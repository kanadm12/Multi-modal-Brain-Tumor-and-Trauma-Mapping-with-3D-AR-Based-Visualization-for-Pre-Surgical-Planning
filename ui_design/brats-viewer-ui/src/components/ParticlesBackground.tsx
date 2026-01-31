"use client";

import React, { useCallback, useEffect, useRef } from "react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import type { Engine } from "tsparticles-engine";

const ParticlesBackground = () => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.playbackRate = 0.5; // Slows down the video to half speed
        }
    }, []);

    const particlesInit = useCallback(async (engine: Engine) => {
        await loadSlim(engine);
    }, []);
    const options = {
        background: {
            color: {
                value: "transparent", // This needs to be transparent to see the video
            },
        },
        fpsLimit: 60,
        interactivity: {
            events: {
                onHover: {
                    enable: true,
                    mode: "grab", // Grabbing a node pulls others with it
                },
                resize: true,
            },
            modes: {
                grab: {
                    distance: 200,
                    links: {
                        opacity: 0.8,
                        color: "#314EE6" // Links become brighter when grabbed
                    }
                },
            },
        },
        particles: {
            color: {
                value: "#ffffff", // Neuron color
            },
            links: {
                color: "#314EE6", // Axon/dendrite color
                distance: 150,
                enable: true,
                opacity: 0.3, // Subtle links
                width: 1,
            },
            collisions: {
                enable: true,
            },
            move: {
                direction: "none" as const,
                enable: true,
                outModes: {
                    default: "bounce" as const,
                },
                random: true, // More organic movement
                speed: 0.5, // Slow drift
                straight: false,
            },
            number: {
                density: {
                    enable: true,
                    area: 800,
                },
                value: 80, // Number of neurons
            },
            opacity: {
                value: 0.4,
            },
            shape: {
                type: "circle",
            },
            size: {
                value: { min: 1, max: 4 }, // Varying neuron sizes
            },
        },
        detectRetina: true,
    };

    return (
        <>
            <video
                ref={videoRef}
                autoPlay
                loop
                muted
                style={{
                    position: 'fixed',
                    width: '100vw',
                    height: '100vh',
                    objectFit: 'cover',
                    zIndex: -2, // Farthest back
                }}
            >
                <source src="/234416.mp4" type="video/mp4" />
            </video>
            <div style={{
                position: 'fixed',
                width: '100%',
                height: '100%',
                backgroundColor: 'rgba(0, 0, 0, 0.3)', // Reduced overlay opacity to make video brighter
                zIndex: -1, // Above video, behind content
            }} />
            <Particles id="tsparticles" init={particlesInit} options={options as any} />
        </>
    );
};

export default ParticlesBackground;