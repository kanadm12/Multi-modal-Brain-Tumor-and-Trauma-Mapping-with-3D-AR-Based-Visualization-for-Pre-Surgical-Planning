"use client";

import React, { useState } from 'react';
import { Box, Button, TextField, Typography, Container, Paper, AppBar, Toolbar } from '@mui/material';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';


export default function CredentialsPage() {
    // ... (your existing state and functions)
    const [credentials, setCredentials] = useState({ name: '', email: '', designation: '', hospital: '' });
    const router = useRouter();

    const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const { name, value } = event.target;
        setCredentials(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = () => {
        if (Object.values(credentials).every(field => field.trim() !== '')) {
            router.push('/upload');
        } else {
            alert("Please fill in all fields.");
        }
    };

    return (
        <Container component="main" maxWidth="sm" sx={{ mt: 8 }}>
            {/* ... (rest of the page content is the same) */}
            <motion.div initial={{ opacity: 0, y: -50 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
                <Paper 
                    elevation={3} 
                    sx={{ 
                        p: 4, 
                        display: 'flex', 
                        flexDirection: 'column', 
                        alignItems: 'center', 
                        borderRadius: 2,
                        border: '2px solid',
                        borderColor: 'primary.main',
                        boxShadow: (theme) => `0 0 12px ${theme.palette.primary.main}`
                    }}
                >
                    <Typography component="h1" variant="h5">Doctor Details</Typography>
                    <Box sx={{ mt: 3, width: '100%' }}>
                        <TextField name="name" label="Name" fullWidth required autoFocus margin="normal" value={credentials.name} onChange={handleInputChange}/>
                        <TextField name="email" label="Email Address" type="email" fullWidth required margin="normal" value={credentials.email} onChange={handleInputChange}/>
                        <TextField name="designation" label="Designation (e.g., Neurosurgeon)" fullWidth required margin="normal" value={credentials.designation} onChange={handleInputChange}/>
                        <TextField name="hospital" label="Hospital/Clinic Name" fullWidth required margin="normal" value={credentials.hospital} onChange={handleInputChange}/>
                        <Button type="submit" fullWidth variant="contained" sx={{ mt: 3, py: 1.5 }} onClick={handleSubmit}>Proceed to Upload</Button>
                    </Box>
                </Paper>
            </motion.div>
        </Container>
    );
}