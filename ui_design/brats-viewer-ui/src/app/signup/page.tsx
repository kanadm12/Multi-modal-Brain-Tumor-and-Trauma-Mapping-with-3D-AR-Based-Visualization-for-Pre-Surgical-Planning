'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Box,
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Link,
  Alert,
  CircularProgress,
  MenuItem,
} from '@mui/material';
import { useAuth } from '@/contexts/AuthContext';

export default function SignupPage() {
  const router = useRouter();
  const { signup, loading } = useAuth();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    full_name: '',
    role: 'doctor',
    hospital: '',
  });
  const [error, setError] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
    setError('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    // Validate passwords match
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    // Validate password strength
    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters long');
      return;
    }

    try {
      await signup({
        email: formData.email,
        password: formData.password,
        full_name: formData.full_name,
        role: formData.role,
        hospital: formData.hospital,
      });
      router.push('/upload'); // Redirect to upload page after signup
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Signup failed. Please try again.');
    }
  };

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          minHeight: '80vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          py: 4,
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 4,
            width: '100%',
            borderRadius: 2,
            background: 'rgba(255, 255, 255, 0.95)',
          }}
        >
          <Typography variant="h4" component="h1" gutterBottom align="center" color="primary">
            Create Account
          </Typography>
          <Typography variant="body2" color="primary" align="center" sx={{ mb: 3 }}>
            Join us to start analyzing brain tumor MRI scans
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Full Name"
              name="full_name"
              value={formData.full_name}
              onChange={handleChange}
              required
              margin="normal"
              autoComplete="name"
              autoFocus
              InputLabelProps={{
                style: { color: '#1976d2' }
              }}
              InputProps={{
                style: { color: '#000' }
              }}
            />
            <TextField
              fullWidth
              label="Email Address"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              required
              margin="normal"
              autoComplete="email"
              InputLabelProps={{
                style: { color: '#1976d2' }
              }}
              InputProps={{
                style: { color: '#000' }
              }}
            />
            <TextField
              fullWidth
              select
              label="Role"
              name="role"
              value={formData.role}
              onChange={handleChange}
              required
              margin="normal"
              InputLabelProps={{
                style: { color: '#1976d2' }
              }}
              InputProps={{
                style: { color: '#000' }
              }}
            >
              <MenuItem value="doctor">Doctor</MenuItem>
              <MenuItem value="radiologist">Radiologist</MenuItem>
              <MenuItem value="researcher">Researcher</MenuItem>
            </TextField>
            <TextField
              fullWidth
              label="Hospital / Institution"
              name="hospital"
              value={formData.hospital}
              onChange={handleChange}
              margin="normal"
              placeholder="e.g., General Hospital"
              InputLabelProps={{
                style: { color: '#1976d2' }
              }}
              InputProps={{
                style: { color: '#000' }
              }}
            />
            <TextField
              fullWidth
              label="Password"
              name="password"
              type="password"
              value={formData.password}
              onChange={handleChange}
              required
              margin="normal"
              autoComplete="new-password"
              helperText="Must be at least 8 characters"
              InputLabelProps={{
                style: { color: '#1976d2' }
              }}
              InputProps={{
                style: { color: '#000' }
              }}
            />
            <TextField
              fullWidth
              label="Confirm Password"
              name="confirmPassword"
              type="password"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
              margin="normal"
              autoComplete="new-password"
              InputLabelProps={{
                style: { color: '#1976d2' }
              }}
              InputProps={{
                style: { color: '#000' }
              }}
            />

            <Button
              type="submit"
              fullWidth
              variant="contained"
              size="large"
              disabled={loading}
              sx={{ mt: 3, mb: 2, py: 1.5 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Create Account'}
            </Button>

            <Box sx={{ textAlign: 'center', mt: 2 }}>
              <Typography variant="body2" color="primary">
                Already have an account?{' '}
                <Link href="/login" underline="hover" color="primary">
                  Sign in here
                </Link>
              </Typography>
            </Box>
          </form>
        </Paper>
      </Box>
    </Container>
  );
}
