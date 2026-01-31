"use client";

import Image from 'next/image';
import Link from 'next/link';

const Logo = () => (
  <Link href="/" style={{ display: 'flex', alignItems: 'center' }}>
    {/* The logo file should be in the `public` folder. */}
    <Image 
      src="/logo.png" 
      alt="Company Logo" 
      width={100}  // You can adjust the width as needed
      height={40} // You can adjust the height as needed
      priority
    />
  </Link>
);

export default Logo;
