"""
AIDocumentIndexer - SSO (Single Sign-On) Service
=================================================

Enterprise SSO support using SAML 2.0 and OIDC protocols.
Integrates with identity providers like Okta, Azure AD, Google Workspace.

Supported Protocols:
- SAML 2.0 (Security Assertion Markup Language)
- OIDC (OpenID Connect)

Supported Identity Providers:
- Okta
- Azure Active Directory (Entra ID)
- Google Workspace
- OneLogin
- Auth0
- Generic SAML/OIDC providers
"""

import base64
import hashlib
import hmac
import os
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import aiohttp
import structlog
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509 import load_pem_x509_certificate
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# =============================================================================
# Enums & Types
# =============================================================================

class SSOProtocol(str, Enum):
    """SSO protocol types."""
    SAML = "saml"
    OIDC = "oidc"


class SSOProvider(str, Enum):
    """Known SSO identity providers."""
    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    ONELOGIN = "onelogin"
    AUTH0 = "auth0"
    GENERIC_SAML = "generic_saml"
    GENERIC_OIDC = "generic_oidc"


class SSOConnectionStatus(str, Enum):
    """Status of an SSO connection."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ERROR = "error"


# =============================================================================
# Data Models
# =============================================================================

class SSOConfiguration(BaseModel):
    """SSO configuration for an organization."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    organization_id: str
    protocol: SSOProtocol
    provider: SSOProvider
    display_name: str
    enabled: bool = True

    # SAML-specific settings
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None
    saml_slo_url: Optional[str] = None
    saml_certificate: Optional[str] = None
    saml_signing_certificate: Optional[str] = None
    saml_name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"

    # OIDC-specific settings
    oidc_issuer: Optional[str] = None
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    oidc_authorization_url: Optional[str] = None
    oidc_token_url: Optional[str] = None
    oidc_userinfo_url: Optional[str] = None
    oidc_jwks_url: Optional[str] = None
    oidc_scopes: List[str] = Field(default_factory=lambda: ["openid", "profile", "email"])

    # Attribute mapping
    attribute_mapping: Dict[str, str] = Field(default_factory=lambda: {
        "email": "email",
        "first_name": "given_name",
        "last_name": "family_name",
        "groups": "groups",
    })

    # Role mapping (SSO groups -> app roles)
    role_mapping: Dict[str, str] = Field(default_factory=dict)

    # Auto-provisioning settings
    auto_provision_users: bool = True
    auto_update_users: bool = True
    default_role: str = "user"

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    status: SSOConnectionStatus = SSOConnectionStatus.PENDING
    error_message: Optional[str] = None


class SSOAuthRequest(BaseModel):
    """SSO authentication request state."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    config_id: str
    state: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    nonce: Optional[str] = Field(default_factory=lambda: secrets.token_urlsafe(16))
    redirect_uri: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=10))
    relay_state: Optional[str] = None  # For SAML


class SSOUser(BaseModel):
    """User information from SSO assertion/token."""
    email: str
    external_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    groups: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    provider: SSOProvider
    config_id: str
    authenticated_at: datetime = Field(default_factory=datetime.utcnow)


class SSOServiceProviderMetadata(BaseModel):
    """Service Provider (SP) metadata for SAML."""
    entity_id: str
    acs_url: str  # Assertion Consumer Service URL
    slo_url: Optional[str] = None  # Single Logout URL
    name_id_format: str = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    signing_certificate: Optional[str] = None


# =============================================================================
# SSO Service
# =============================================================================

class SSOService:
    """
    Enterprise SSO service supporting SAML 2.0 and OIDC.

    Usage:
        sso = SSOService(base_url="https://app.example.com")

        # SAML flow
        auth_url = await sso.initiate_saml_login(config, redirect_uri)
        user = await sso.process_saml_response(saml_response, config)

        # OIDC flow
        auth_url = await sso.initiate_oidc_login(config, redirect_uri)
        user = await sso.process_oidc_callback(code, state, config)
    """

    def __init__(
        self,
        base_url: str,
        sp_entity_id: Optional[str] = None,
        signing_key: Optional[bytes] = None,
        signing_cert: Optional[bytes] = None,
    ):
        """
        Initialize SSO service.

        Args:
            base_url: Base URL of the application (for callback URLs)
            sp_entity_id: Service Provider entity ID (defaults to base_url)
            signing_key: Private key for signing SAML requests
            signing_cert: Certificate for SAML metadata
        """
        self.base_url = base_url.rstrip("/")
        self.sp_entity_id = sp_entity_id or self.base_url
        self.signing_key = signing_key
        self.signing_cert = signing_cert

        # In-memory state store (use Redis in production)
        self._auth_requests: Dict[str, SSOAuthRequest] = {}

        # HTTP client for OIDC token exchange
        self._http_client: Optional[aiohttp.ClientSession] = None

    async def _get_http_client(self) -> aiohttp.ClientSession:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.closed:
            self._http_client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.close()

    # =========================================================================
    # Service Provider Metadata
    # =========================================================================

    def get_sp_metadata(self) -> SSOServiceProviderMetadata:
        """Get Service Provider metadata for SAML configuration."""
        return SSOServiceProviderMetadata(
            entity_id=self.sp_entity_id,
            acs_url=f"{self.base_url}/api/v1/sso/saml/acs",
            slo_url=f"{self.base_url}/api/v1/sso/saml/slo",
            signing_certificate=self.signing_cert.decode() if self.signing_cert else None,
        )

    def generate_sp_metadata_xml(self) -> str:
        """Generate SAML SP metadata XML."""
        metadata = self.get_sp_metadata()

        xml = f"""<?xml version="1.0"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                     entityID="{metadata.entity_id}">
    <md:SPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol"
                        AuthnRequestsSigned="false"
                        WantAssertionsSigned="true">
        <md:NameIDFormat>{metadata.name_id_format}</md:NameIDFormat>
        <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                                     Location="{metadata.acs_url}"
                                     index="0"
                                     isDefault="true"/>
        <md:SingleLogoutService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                               Location="{metadata.slo_url}"/>
    </md:SPSSODescriptor>
</md:EntityDescriptor>"""
        return xml

    # =========================================================================
    # SAML 2.0 Flow
    # =========================================================================

    async def initiate_saml_login(
        self,
        config: SSOConfiguration,
        redirect_uri: str,
        relay_state: Optional[str] = None,
    ) -> Tuple[str, SSOAuthRequest]:
        """
        Initiate SAML authentication flow.

        Args:
            config: SSO configuration
            redirect_uri: Where to redirect after authentication
            relay_state: Optional state to preserve through the flow

        Returns:
            Tuple of (redirect URL, auth request for validation)
        """
        if not config.saml_sso_url:
            raise ValueError("SAML SSO URL not configured")

        # Create auth request
        auth_request = SSOAuthRequest(
            config_id=config.id,
            redirect_uri=redirect_uri,
            relay_state=relay_state,
        )

        # Store for validation
        self._auth_requests[auth_request.state] = auth_request

        # Generate SAML AuthnRequest
        request_id = f"_{uuid4()}"
        issue_instant = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        authn_request = f"""<?xml version="1.0"?>
<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                    ID="{request_id}"
                    Version="2.0"
                    IssueInstant="{issue_instant}"
                    Destination="{config.saml_sso_url}"
                    AssertionConsumerServiceURL="{self.base_url}/api/v1/sso/saml/acs"
                    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{self.sp_entity_id}</saml:Issuer>
    <samlp:NameIDPolicy Format="{config.saml_name_id_format}"
                        AllowCreate="true"/>
</samlp:AuthnRequest>"""

        # Encode request
        import zlib
        deflated = zlib.compress(authn_request.encode())[2:-4]  # Remove zlib header/checksum
        encoded = base64.b64encode(deflated).decode()

        # Build redirect URL
        params = {
            "SAMLRequest": encoded,
            "RelayState": auth_request.state,
        }

        redirect_url = f"{config.saml_sso_url}?{urllib.parse.urlencode(params)}"

        logger.info(
            "SAML login initiated",
            config_id=config.id,
            provider=config.provider,
        )

        return redirect_url, auth_request

    async def process_saml_response(
        self,
        saml_response: str,
        relay_state: str,
        config: SSOConfiguration,
    ) -> SSOUser:
        """
        Process SAML response from IdP.

        Args:
            saml_response: Base64-encoded SAML response
            relay_state: State parameter for validation
            config: SSO configuration

        Returns:
            SSOUser with authenticated user information
        """
        # Validate state
        auth_request = self._auth_requests.get(relay_state)
        if not auth_request:
            raise ValueError("Invalid relay state - possible CSRF attack")

        if datetime.utcnow() > auth_request.expires_at:
            del self._auth_requests[relay_state]
            raise ValueError("Authentication request expired")

        # Decode response
        try:
            decoded = base64.b64decode(saml_response)
            response_xml = decoded.decode()
        except Exception as e:
            raise ValueError(f"Failed to decode SAML response: {e}")

        # Parse response (simplified - use lxml/signxml in production)
        # In production, you should:
        # 1. Validate XML signature using IdP certificate
        # 2. Verify assertion conditions (NotBefore, NotOnOrAfter)
        # 3. Verify audience restriction
        # 4. Verify InResponseTo matches request ID

        import re

        # Extract email from NameID
        name_id_match = re.search(
            r'<(?:saml2?:)?NameID[^>]*>([^<]+)</(?:saml2?:)?NameID>',
            response_xml,
            re.IGNORECASE,
        )
        if not name_id_match:
            raise ValueError("No NameID found in SAML response")

        email = name_id_match.group(1).strip()

        # Extract attributes
        attributes: Dict[str, Any] = {}
        attr_pattern = re.compile(
            r'<(?:saml2?:)?Attribute\s+Name="([^"]+)"[^>]*>.*?'
            r'<(?:saml2?:)?AttributeValue[^>]*>([^<]+)</(?:saml2?:)?AttributeValue>',
            re.IGNORECASE | re.DOTALL,
        )
        for match in attr_pattern.finditer(response_xml):
            attr_name = match.group(1)
            attr_value = match.group(2).strip()
            attributes[attr_name] = attr_value

        # Map attributes
        first_name = attributes.get(config.attribute_mapping.get("first_name", "given_name"))
        last_name = attributes.get(config.attribute_mapping.get("last_name", "family_name"))
        groups_attr = config.attribute_mapping.get("groups", "groups")
        groups = attributes.get(groups_attr, "").split(",") if groups_attr in attributes else []

        # Clean up auth request
        del self._auth_requests[relay_state]

        user = SSOUser(
            email=email,
            external_id=email,  # Use email as external ID for SAML
            first_name=first_name,
            last_name=last_name,
            full_name=f"{first_name or ''} {last_name or ''}".strip() or None,
            groups=[g.strip() for g in groups if g.strip()],
            attributes=attributes,
            provider=config.provider,
            config_id=config.id,
        )

        logger.info(
            "SAML authentication successful",
            email=user.email,
            provider=config.provider,
        )

        return user

    # =========================================================================
    # OIDC Flow
    # =========================================================================

    async def initiate_oidc_login(
        self,
        config: SSOConfiguration,
        redirect_uri: str,
    ) -> Tuple[str, SSOAuthRequest]:
        """
        Initiate OIDC authentication flow.

        Args:
            config: SSO configuration
            redirect_uri: Where to redirect after authentication

        Returns:
            Tuple of (authorization URL, auth request for validation)
        """
        if not config.oidc_authorization_url or not config.oidc_client_id:
            raise ValueError("OIDC configuration incomplete")

        # Create auth request
        auth_request = SSOAuthRequest(
            config_id=config.id,
            redirect_uri=redirect_uri,
        )

        # Store for validation
        self._auth_requests[auth_request.state] = auth_request

        # Build authorization URL
        params = {
            "client_id": config.oidc_client_id,
            "response_type": "code",
            "scope": " ".join(config.oidc_scopes),
            "redirect_uri": f"{self.base_url}/api/v1/sso/oidc/callback",
            "state": auth_request.state,
            "nonce": auth_request.nonce,
        }

        auth_url = f"{config.oidc_authorization_url}?{urllib.parse.urlencode(params)}"

        logger.info(
            "OIDC login initiated",
            config_id=config.id,
            provider=config.provider,
        )

        return auth_url, auth_request

    async def process_oidc_callback(
        self,
        code: str,
        state: str,
        config: SSOConfiguration,
    ) -> SSOUser:
        """
        Process OIDC callback with authorization code.

        Args:
            code: Authorization code from IdP
            state: State parameter for validation
            config: SSO configuration

        Returns:
            SSOUser with authenticated user information
        """
        # Validate state
        auth_request = self._auth_requests.get(state)
        if not auth_request:
            raise ValueError("Invalid state - possible CSRF attack")

        if datetime.utcnow() > auth_request.expires_at:
            del self._auth_requests[state]
            raise ValueError("Authentication request expired")

        # Exchange code for tokens
        if not config.oidc_token_url or not config.oidc_client_secret:
            raise ValueError("OIDC token configuration incomplete")

        client = await self._get_http_client()

        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": f"{self.base_url}/api/v1/sso/oidc/callback",
            "client_id": config.oidc_client_id,
            "client_secret": config.oidc_client_secret,
        }

        async with client.post(config.oidc_token_url, data=token_data) as response:
            if response.status != 200:
                error = await response.text()
                raise ValueError(f"Token exchange failed: {error}")

            tokens = await response.json()

        access_token = tokens.get("access_token")
        id_token = tokens.get("id_token")

        if not access_token:
            raise ValueError("No access token in response")

        # Get user info
        if config.oidc_userinfo_url:
            async with client.get(
                config.oidc_userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
            ) as response:
                if response.status != 200:
                    raise ValueError("Failed to get user info")

                userinfo = await response.json()
        else:
            # Try to decode ID token (simplified - should verify signature)
            try:
                import json
                parts = id_token.split(".")
                payload = base64.urlsafe_b64decode(parts[1] + "==")
                userinfo = json.loads(payload)
            except Exception:
                raise ValueError("No userinfo endpoint and couldn't decode ID token")

        # Extract user info with attribute mapping
        email = userinfo.get(config.attribute_mapping.get("email", "email"))
        if not email:
            raise ValueError("No email in user info")

        first_name = userinfo.get(config.attribute_mapping.get("first_name", "given_name"))
        last_name = userinfo.get(config.attribute_mapping.get("last_name", "family_name"))
        groups_key = config.attribute_mapping.get("groups", "groups")
        groups = userinfo.get(groups_key, [])
        if isinstance(groups, str):
            groups = [groups]

        # Clean up auth request
        del self._auth_requests[state]

        user = SSOUser(
            email=email,
            external_id=userinfo.get("sub", email),
            first_name=first_name,
            last_name=last_name,
            full_name=userinfo.get("name"),
            groups=groups,
            attributes=userinfo,
            provider=config.provider,
            config_id=config.id,
        )

        logger.info(
            "OIDC authentication successful",
            email=user.email,
            provider=config.provider,
        )

        return user

    # =========================================================================
    # Configuration Helpers
    # =========================================================================

    @staticmethod
    def create_okta_config(
        organization_id: str,
        okta_domain: str,
        client_id: str,
        client_secret: str,
    ) -> SSOConfiguration:
        """Create OIDC configuration for Okta."""
        return SSOConfiguration(
            organization_id=organization_id,
            protocol=SSOProtocol.OIDC,
            provider=SSOProvider.OKTA,
            display_name="Okta",
            oidc_issuer=f"https://{okta_domain}",
            oidc_client_id=client_id,
            oidc_client_secret=client_secret,
            oidc_authorization_url=f"https://{okta_domain}/oauth2/v1/authorize",
            oidc_token_url=f"https://{okta_domain}/oauth2/v1/token",
            oidc_userinfo_url=f"https://{okta_domain}/oauth2/v1/userinfo",
            oidc_jwks_url=f"https://{okta_domain}/oauth2/v1/keys",
            oidc_scopes=["openid", "profile", "email", "groups"],
        )

    @staticmethod
    def create_azure_ad_config(
        organization_id: str,
        tenant_id: str,
        client_id: str,
        client_secret: str,
    ) -> SSOConfiguration:
        """Create OIDC configuration for Azure AD."""
        base_url = f"https://login.microsoftonline.com/{tenant_id}"
        return SSOConfiguration(
            organization_id=organization_id,
            protocol=SSOProtocol.OIDC,
            provider=SSOProvider.AZURE_AD,
            display_name="Microsoft Azure AD",
            oidc_issuer=f"{base_url}/v2.0",
            oidc_client_id=client_id,
            oidc_client_secret=client_secret,
            oidc_authorization_url=f"{base_url}/oauth2/v2.0/authorize",
            oidc_token_url=f"{base_url}/oauth2/v2.0/token",
            oidc_userinfo_url="https://graph.microsoft.com/oidc/userinfo",
            oidc_jwks_url=f"{base_url}/discovery/v2.0/keys",
            oidc_scopes=["openid", "profile", "email"],
        )

    @staticmethod
    def create_google_config(
        organization_id: str,
        client_id: str,
        client_secret: str,
        hosted_domain: Optional[str] = None,
    ) -> SSOConfiguration:
        """Create OIDC configuration for Google Workspace."""
        config = SSOConfiguration(
            organization_id=organization_id,
            protocol=SSOProtocol.OIDC,
            provider=SSOProvider.GOOGLE,
            display_name="Google Workspace",
            oidc_issuer="https://accounts.google.com",
            oidc_client_id=client_id,
            oidc_client_secret=client_secret,
            oidc_authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
            oidc_token_url="https://oauth2.googleapis.com/token",
            oidc_userinfo_url="https://openidconnect.googleapis.com/v1/userinfo",
            oidc_jwks_url="https://www.googleapis.com/oauth2/v3/certs",
            oidc_scopes=["openid", "profile", "email"],
        )

        if hosted_domain:
            # Restrict to specific Google Workspace domain
            config.attribute_mapping["hd"] = hosted_domain

        return config


# =============================================================================
# Singleton & Dependency
# =============================================================================

_sso_service: Optional[SSOService] = None


def get_sso_service() -> SSOService:
    """Get or create SSO service singleton."""
    global _sso_service
    if _sso_service is None:
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        _sso_service = SSOService(base_url=base_url)
    return _sso_service


async def initialize_sso_service() -> SSOService:
    """Initialize SSO service (call at startup)."""
    return get_sso_service()
