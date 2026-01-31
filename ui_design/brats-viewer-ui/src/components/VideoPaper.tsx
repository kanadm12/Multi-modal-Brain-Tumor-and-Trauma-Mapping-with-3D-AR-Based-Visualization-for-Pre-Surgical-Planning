import React from 'react';
import { Paper, PaperProps, Box } from '@mui/material';

interface VideoPaperProps extends PaperProps {
  videoSrc?: string;
}

const VideoPaper: React.FC<React.PropsWithChildren<VideoPaperProps>> = ({
  children,
  videoSrc = '/234416.mp4',
  sx,
  ...props
}) => {
  return (
    <Paper
      sx={{
        position: 'relative',
        overflow: 'hidden',
        // Ensure the Paper's direct background is transparent to see the video
        backgroundColor: 'transparent',
        ...sx,
      }}
      {...props}
    >
      <Box
        component="video"
        autoPlay
        loop
        muted
        src={videoSrc}
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          zIndex: -1, // Place video behind the content
          opacity: 0.15, // Adjust for desired video visibility
        }}
      />
      {/* Content will be placed above the video */}
      {children}
    </Paper>
  );
};

export default VideoPaper;