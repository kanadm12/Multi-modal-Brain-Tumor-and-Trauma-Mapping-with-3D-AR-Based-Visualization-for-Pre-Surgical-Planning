// src/theme.ts
import { createTheme } from '@mui/material/styles';

const PRIMARY_BLUE = '#314EE6';
const BACKGROUND_DEFAULT = 'transparent';
const BACKGROUND_PAPER = '#1a2a3a'; // Dark, slightly blue paper background
const TEXT_PRIMARY = '#e6f1ff'; // A very light blue, almost white
const TEXT_SECONDARY = 'rgba(230, 241, 255, 0.7)';

export const brandTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: PRIMARY_BLUE,
    },
    background: {
      default: BACKGROUND_DEFAULT,
      paper: BACKGROUND_PAPER,
    },
    text: {
      primary: TEXT_PRIMARY,
      secondary: TEXT_SECONDARY,
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: 'transparent',
          backgroundImage: 'none',
          boxShadow: 'none',
          borderBottom: `1px solid rgba(49, 78, 230, 0.3)`,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 'bold',
        },
        contained: {
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: 'rgba(49, 78, 230, 0.4)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(49, 78, 230, 0.8)',
            },
            '&.Mui-focused fieldset': {
              borderColor: PRIMARY_BLUE,
            },
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          position: 'relative',
          // This will be the fallback for standard Paper, and a tint for VideoPaper
          backgroundColor: 'rgba(26, 42, 58, 0.8)',
          boxShadow: `5px 5px 10px #060e15, -5px -5px 10px #1c3855, inset 1px 1px 2px rgba(49, 78, 230, 0.3)`,
        },
      },
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    allVariants: {
      color: TEXT_PRIMARY,
      textShadow: `0px 0px 8px rgba(49, 78, 230, 0.5)`,
    },
    h1: { fontWeight: 'bold' },
    h2: { fontWeight: 'bold' },
    h3: { fontWeight: 'bold' },
    h4: { fontWeight: 'bold' },
    h5: { fontWeight: 'bold' },
    h6: { fontWeight: 'bold' },
  },
});
