"use client";

import React, { useEffect, useRef } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import type { ISourceOptions } from "@tsparticles/engine";
import { loadSlim } from "@tsparticles/slim";

const ParticlesBackground = () => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useEffect(() => {
        if (videoRef.current) {
            videoRef.current.playbackRate = 0.5;
        }

        initParticlesEngine(async (engine) => {
            await loadSlim(engine);
        });
    }, []);

    const options = {
        background: {
            color: {
                value: "transparent",
            },
        },
        fpsLimit: 60,
        interactivity: {
            events: {
                onHover: {
                    enable: true,
                    mode: "grab",
                },
                resize: true,
            },
            modes: {
                grab: {
                    distance: 200,
                    links: {
                        opacity: 0.8,
                        color: "#314EE6",
                    }
                },
            },
        },
        particles: {
            color: {
                value: "#ffffff",
            },
            links: {
                color: "#314EE6",
                distance: 150,
                enable: true,
                opacity: 0.3,
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
                random: true,
                speed: 0.5,
                straight: false,
            },
            number: {
                density: {
                    enable: true,
                    area: 800,
                },
                value: 80,
            },
            opacity: {
                value: 0.4,
            },
            shape: {
                type: "circle",
            },
            size: {
                value: { min: 1, max: 4 },
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
                    position: "fixed",
                    width: "100vw",
                    height: "100vh",
                    objectFit: "cover",
                    zIndex: -2,
                }}
            >
                <source src="/234416.mp4" type="video/mp4" />
            </video>
            <div
                style={{
                    position: "fixed",
                    width: "100%",
                    height: "100%",
                    backgroundColor: "rgba(0, 0, 0, 0.3)",
                    zIndex: -1,
                }}
            />
            <Particles id="tsparticles" options={options as ISourceOptions} />
        </>
    );
};

export default ParticlesBackground;