/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',

  experimental: {
    serverActions: {
      bodySizeLimit: '50mb',
    },
  },

  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'localhost',
      },
      {
        protocol: 'http',
        hostname: 'localhost',
      },
    ],
  },

  async rewrites() {
    // Strip /api/v1 suffix if present to get the base backend URL
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const backendBase = apiUrl.replace(/\/api\/v1\/?$/, '');
    return [
      {
        source: '/api/backend/:path*',
        destination: `${backendBase}/:path*`,
      },
      {
        source: '/api/v1/:path*',
        destination: `${backendBase}/api/v1/:path*`,
      },
    ];
  },

  webpack: (config, { isServer }) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
    };

    // Fix pdfjs-dist ESM import and canvas module for SSR
    // See: https://github.com/vercel/next.js/issues/58313
    if (isServer) {
      config.resolve.alias = {
        ...config.resolve.alias,
        canvas: false,
      };
    }

    // Ensure pdfjs-dist works with dynamic imports
    config.resolve.alias = {
      ...config.resolve.alias,
      'pdfjs-dist': 'pdfjs-dist/build/pdf.mjs',
    };

    return config;
  },
};

module.exports = nextConfig;
