/**
 * NextAuth.js v5 Configuration
 * ============================
 *
 * Authentication configuration for NextAuth v5 (Auth.js).
 */

import NextAuth from 'next-auth';
import Credentials from 'next-auth/providers/credentials';
import Google from 'next-auth/providers/google';

// API base URL for backend authentication
const API_BASE_URL = process.env.BACKEND_URL || 'http://localhost:8000/api/v1';

// Helper to decode JWT and check expiration
function isTokenExpired(token: string | null | undefined): boolean {
  if (!token) return true;
  try {
    const payload = JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString());
    // Token is expired if less than 5 minutes remaining
    const expiresAt = payload.exp * 1000;
    const fiveMinutes = 5 * 60 * 1000;
    return Date.now() > expiresAt - fiveMinutes;
  } catch {
    return true;
  }
}

// Helper to refresh token from backend
async function refreshAccessToken(token: any): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token.accessToken}`,
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error('Token refresh failed');
    }

    const data = await response.json();
    console.log('[Auth] Token refreshed successfully');

    return {
      ...token,
      accessToken: data.access_token,
      error: undefined,
    };
  } catch (error) {
    console.error('[Auth] Failed to refresh token:', error);
    return {
      ...token,
      error: 'RefreshAccessTokenError',
    };
  }
}

export const { handlers, signIn, signOut, auth } = NextAuth({
  providers: [
    // Credentials provider for email/password login
    Credentials({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        try {
          // Authenticate against backend API
          const response = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              email: credentials.email,
              password: credentials.password,
            }),
          });

          if (!response.ok) {
            return null;
          }

          const data = await response.json();

          // Return user object with token
          return {
            id: data.user.id,
            email: data.user.email,
            name: data.user.full_name,
            role: data.user.role,
            accessTier: data.user.access_tier,
            accessToken: data.access_token,
          };
        } catch (error) {
          // For development, allow mock login
          if (process.env.NODE_ENV === 'development' && credentials.email === 'admin@example.com') {
            return {
              id: '550e8400-e29b-41d4-a716-446655440000',
              email: 'admin@example.com',
              name: 'Admin User',
              role: 'admin',
              accessTier: 100,
              accessToken: 'dev-token-' + Date.now(),
            };
          }
          return null;
        }
      },
    }),

    // Google OAuth provider (optional)
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET
      ? [
          Google({
            clientId: process.env.GOOGLE_CLIENT_ID,
            clientSecret: process.env.GOOGLE_CLIENT_SECRET,
          }),
        ]
      : []),
  ],

  callbacks: {
    async jwt({ token, user }) {
      // Initial sign in
      if (user) {
        token.id = user.id;
        token.role = (user as any).role;
        token.accessTier = (user as any).accessTier;
        token.accessToken = (user as any).accessToken;
        return token;
      }

      // Check if token needs refresh
      if (token.accessToken && isTokenExpired(token.accessToken as string)) {
        console.log('[Auth] Token expired or expiring soon, attempting refresh...');
        return await refreshAccessToken(token);
      }

      return token;
    },

    async session({ session, token }) {
      // Add custom properties to session
      if (session.user) {
        (session.user as any).id = token.id as string;
        (session.user as any).role = token.role as string;
        (session.user as any).accessTier = token.accessTier as number;
      }
      (session as any).accessToken = token.accessToken as string;
      // Pass through any error from token refresh
      (session as any).error = token.error;
      return session;
    },
  },

  pages: {
    signIn: '/login',
    error: '/login',
  },

  session: {
    strategy: 'jwt',
    maxAge: 24 * 60 * 60, // 24 hours
  },

  trustHost: true,

  debug: process.env.NODE_ENV === 'development',
});
