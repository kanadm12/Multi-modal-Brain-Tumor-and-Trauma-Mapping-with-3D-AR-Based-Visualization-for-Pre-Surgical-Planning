// src/app/layout.tsx
"use client";

import React, { useState, useEffect } from 'react';
import { AppBar, Box, Toolbar, Button, Typography } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { brandTheme } from '@/theme'; // Import your new theme
import ParticlesBackground from '@/components/ParticlesBackground';
import Logo from '@/components/Logo';
import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import LogoutIcon from '@mui/icons-material/Logout';
import { useRouter } from 'next/navigation';

function HeaderContent() {
  const { isAuthenticated, user, logout } = useAuth();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const handleLogout = async () => {
    await logout();
    router.push('/');
  };

  return (
    <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: '1px solid rgba(49, 78, 230, 0.3)' }}>
      <Toolbar sx={{ justifyContent: 'space-between', py: 1, px: 3 }}>
        <Logo />
        {mounted && isAuthenticated && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" color="text.secondary">
              {user?.full_name}
            </Typography>
            <Button
              variant="outlined"
              size="small"
              startIcon={<LogoutIcon />}
              onClick={handleLogout}
              sx={{ borderRadius: 2 }}
            >
              Logout
            </Button>
          </Box>
        )}
      </Toolbar>
    </AppBar>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>
        <ThemeProvider theme={brandTheme}>
          <AuthProvider>
            <ParticlesBackground />
            <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', position: 'relative', zIndex: 1, bgcolor: 'background.default' }}>
              <HeaderContent />
              <Box component="main" sx={{ flexGrow: 1, position: 'relative' }}>
                {children}
              </Box>
            </Box>
          </AuthProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
