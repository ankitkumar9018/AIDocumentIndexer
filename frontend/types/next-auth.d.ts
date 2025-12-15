/**
 * NextAuth.js Type Declarations
 * ==============================
 *
 * Extends NextAuth types with custom user properties.
 */

import 'next-auth';
import { JWT } from 'next-auth/jwt';

declare module 'next-auth' {
  interface User {
    id: string;
    email: string;
    name: string;
    role: string;
    accessTier: number;
    accessToken: string;
  }

  interface Session {
    user: {
      id: string;
      email: string;
      name: string;
      role: string;
      accessTier: number;
      image?: string;
    };
    accessToken: string;
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    id: string;
    role: string;
    accessTier: number;
    accessToken: string;
  }
}
