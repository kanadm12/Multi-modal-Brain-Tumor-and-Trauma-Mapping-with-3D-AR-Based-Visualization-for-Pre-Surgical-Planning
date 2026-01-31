// src/app/layout.tsx
"use client";

import React from 'react';
import { AppBar, Box, Toolbar } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { brandTheme } from '@/theme'; // Import your new theme
import ParticlesBackground from '@/components/ParticlesBackground';
import Logo from '@/components/Logo';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider theme={brandTheme}>
          <ParticlesBackground />
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1, bgcolor: 'background.default' }}>
            <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid rgba(49, 78, 230, 0.3)' }}>
              <Toolbar sx={{ justifyContent: 'flex-start', py: 1, px: 3 }}>
                  <Logo />
              </Toolbar>
            </AppBar>
            <Box component="main" sx={{ flexGrow: 1, position: 'relative' }}>
              {children}
            </Box>
          </Box>
        </ThemeProvider>
      </body>
    </html>
  );
}
