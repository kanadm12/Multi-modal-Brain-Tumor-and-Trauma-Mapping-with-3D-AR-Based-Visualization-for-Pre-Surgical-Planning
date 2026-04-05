import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Image optimization (using remotePatterns instead of deprecated domains)
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
    unoptimized: true,
  },

  // Turbopack configuration (Next.js 16+ default)
  turbopack: {
    rules: {
      '*.glb': {
        loaders: ['raw-loader'],
        as: '*.js',
      },
      '*.gltf': {
        loaders: ['raw-loader'],
        as: '*.js',
      },
    },
  },

  // Ignore TypeScript errors during build (optional - remove in strict mode)
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
