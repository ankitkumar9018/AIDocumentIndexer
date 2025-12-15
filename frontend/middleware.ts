/**
 * Next.js Middleware for Route Protection
 * =========================================
 *
 * Protects routes that require authentication using NextAuth v5.
 */

import { auth } from '@/auth';
import { NextResponse } from 'next/server';

export default auth((req) => {
  const { pathname } = req.nextUrl;
  const isLoggedIn = !!req.auth;

  // Redirect unauthenticated users to login
  if (!isLoggedIn) {
    return NextResponse.redirect(new URL('/login', req.url));
  }

  // Admin routes require admin role
  if (pathname.startsWith('/dashboard/admin')) {
    const userRole = (req.auth?.user as any)?.role;
    if (userRole !== 'admin') {
      return NextResponse.redirect(new URL('/dashboard', req.url));
    }
  }

  return NextResponse.next();
});

// Protect all dashboard routes
export const config = {
  matcher: ['/dashboard/:path*'],
};
