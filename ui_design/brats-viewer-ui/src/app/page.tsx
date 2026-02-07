"use client";

import React, { useState, useEffect } from 'react';
import { Box, Button, Typography, Container, Paper, Stack } from '@mui/material';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import LoginIcon from '@mui/icons-material/Login';
import PersonAddIcon from '@mui/icons-material/PersonAdd';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

export default function HomePage() {
    const router = useRouter();
    const { isAuthenticated, user } = useAuth();
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
    }, []);

    return (
        <Container component="main" maxWidth="md" sx={{ mt: 8 }}>
            <motion.div 
                initial={{ opacity: 0, y: -50 }} 
                animate={{ opacity: 1, y: 0 }} 
                transition={{ duration: 0.7 }}
            >
                <Paper 
                    elevation={3} 
                    sx={{ 
                        p: 6, 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center', 
                        borderRadius: 2,
                        border: '2px solid',
                        borderColor: 'primary.main',
                        boxShadow: (theme) => `0 0 12px ${theme.palette.primary.main}`,
                        background: 'rgba(255, 255, 255, 0.95)',
                    }}
                >
                    <Typography 
                        component="h1" 
                        variant="h3" 
                        color="primary"
                        gutterBottom
                        align="center"
                    >
                        BraTS Brain Tumor Analysis
                    </Typography>
                    
                    <Typography 
                        variant="h6" 
                        color="primary" 
                        align="center" 
                        sx={{ mb: 4, maxWidth: 600 }}
                    >
                        Advanced multi-modal brain tumor segmentation with 3D AR visualization
                        for pre-surgical planning
                    </Typography>

                    {mounted && (
                        <>
                            {isAuthenticated ? (
                                <>
                                    <Typography variant="h5" gutterBottom>
                                        Welcome back, {user?.full_name}!
                                    </Typography>
                                    <Button
                                        variant="contained"
                                        size="large"
                                        startIcon={<CloudUploadIcon />}
                                        onClick={() => router.push('/upload')}
                                        sx={{ mt: 3, py: 1.5, px: 4 }}
                                    >
                                        Go to Upload
                                    </Button>
                                </>
                            ) : (
                                <Stack spacing={3} sx={{ mt: 2, width: '100%', maxWidth: 400 }}>
                                    <Button
                                        variant="contained"
                                        size="large"
                                        startIcon={<LoginIcon />}
                                        onClick={() => router.push('/login')}
                                        sx={{ py: 1.5 }}
                                    >
                                        Sign In
                                    </Button>
                                    <Button
                                        variant="outlined"
                                        size="large"
                                        startIcon={<PersonAddIcon />}
                                        onClick={() => router.push('/signup')}
                                        sx={{ py: 1.5 }}
                                    >
                                        Create Account
                                    </Button>
                                </Stack>
                            )}
                        </>
                    )}
                </Paper>
            </motion.div>
        </Container>
    );
}