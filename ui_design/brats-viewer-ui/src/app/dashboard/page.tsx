"use client";

import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  CircularProgress,
  Alert
} from '@mui/material';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import ProtectedRoute from '@/components/ProtectedRoute';
import { apiService, SessionResponse } from '@/services/api';
import VisibilityIcon from '@mui/icons-material/Visibility';
import DownloadIcon from '@mui/icons-material/Download';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

function DashboardPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [sessions, setSessions] = useState<SessionResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const data = await apiService.getUserSessions();
      setSessions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
    } finally {
      setLoading(false);
    }
  };

  const handleViewReport = (sessionId: string) => {
    router.push(`/viewer?session=${sessionId}`);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '50vh' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ color: 'primary.main', fontWeight: 'bold' }}>
          Analysis History
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Welcome back, {user?.full_name}! View your previous brain tumor analyses.
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {sessions.length === 0 ? (
        <Card sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No analyses found
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Upload your first MRI scan to get started
          </Typography>
          <Button
            variant="contained"
            onClick={() => router.push('/upload')}
            sx={{ background: 'linear-gradient(45deg, #2c5aa0 30%, #4CAF50 90%)' }}
          >
            Upload New Scan
          </Button>
        </Card>
      ) : (
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' }, gap: 3 }}>
          {sessions.map((session) => (
            <Card
              key={session.session_id}
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                borderLeft: '4px solid',
                borderColor: session.status === 'completed' ? 'success.main' : 'warning.main',
                '&:hover': {
                  boxShadow: 6,
                  transform: 'translateY(-2px)',
                  transition: 'all 0.3s'
                }
              }}
            >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                    <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
                      {session.patient_name || 'Anonymous Patient'}
                    </Typography>
                    <Chip
                      label={session.status}
                      color={getStatusColor(session.status) as "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning"}
                      size="small"
                      sx={{ textTransform: 'capitalize' }}
                    />
                  </Box>
                  
                  {session.patient_age && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Age: {session.patient_age} years
                    </Typography>
                  )}
                  
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 2, color: 'text.secondary' }}>
                    <AccessTimeIcon sx={{ fontSize: 16, mr: 0.5 }} />
                    <Typography variant="caption">
                      {formatDate(session.created_at)}
                    </Typography>
                  </Box>
                  
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                    Session ID: {session.session_id}
                  </Typography>
                </CardContent>
                
                <CardActions sx={{ p: 2, pt: 0 }}>
                  <Button
                    size="small"
                    startIcon={<VisibilityIcon />}
                    onClick={() => handleViewReport(session.session_id)}
                    disabled={session.status !== 'completed'}
                    sx={{ mr: 1 }}
                  >
                    View Report
                  </Button>
                  {session.has_report && (
                    <Button
                      size="small"
                      startIcon={<DownloadIcon />}
                      onClick={async () => {
                        try {
                          const blob = await apiService.downloadReportPDF(session.session_id);
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `report_${session.session_id}.pdf`;
                          a.click();
                        } catch (_err) {
                          alert('Failed to download PDF');
                        }
                      }}
                    >
                      Download PDF
                    </Button>
                  )}
                </CardActions>
            </Card>
          ))}
        </Box>
      )}
    </Container>
  );
}

export default function DashboardPageWrapper() {
  return (
    <ProtectedRoute>
      <DashboardPage />
    </ProtectedRoute>
  );
}
