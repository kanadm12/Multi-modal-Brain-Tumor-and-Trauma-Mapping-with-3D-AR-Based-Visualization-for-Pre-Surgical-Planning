import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Environment variables exposed to the browser
  env: {
    NEXT_PUBLIC_RUNPOD_ENDPOINT_ID: process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID,
    NEXT_PUBLIC_RUNPOD_API_KEY: process.env.NEXT_PUBLIC_RUNPOD_API_KEY,
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  
  // Image optimization
  images: {
    domains: ['localhost'],
    unoptimized: true,
  },
  
  // Webpack configuration for Three.js and GLTF
  webpack: (config) => {
    config.module.rules.push({
      test: /\.(glb|gltf)$/,
      type: 'asset/resource',
    });
    return config;
  },
};

export default nextConfig;
