# Multi-modal Brain Tumor and Trauma Mapping with 3D AR-Based Visualization for Pre-Surgical Planning

This project is an advanced visualization tool designed for neurosurgeons and medical professionals to assist in pre-surgical planning. By integrating multi-modal brain imaging data (such as MRI, CT, and DTI), the application generates interactive 3D models of the brain, clearly mapping tumors and trauma areas.

The core feature is an Augmented Reality (AR) interface that overlays these 3D models onto the real world, providing an intuitive and immersive planning experience.

## ✨ Key Features

- **Multi-modal Data Fusion:** Combines data from various imaging sources (MRI, CT, etc.) to create a comprehensive 3D brain model.
- **Interactive 3D Visualization:** Allows users to rotate, zoom, and inspect the 3D model of the brain, tumor, and surrounding tissues.
- **Tumor & Trauma Segmentation:** Automatically or semi-automatically segments and highlights pathological areas.
- **Augmented Reality (AR) Overlay:** Projects the 3D model into a real-world view, aiding spatial understanding and surgical approach planning.
- **Web-Based and Accessible:** Built with modern web technologies to be accessible on a range of devices without requiring native installation.

## 🛠️ Technology Stack

- **Framework:** Next.js
- **Language:** TypeScript
- **UI:** React
- **Styling:** Tailwind CSS
- **3D Rendering:** (e.g., Three.js, React Three Fiber)

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Make sure you have Node.js (version 18.x or later) and a package manager (npm, yarn, or pnpm) installed.

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/kanadm12/Multi-modal-Brain-Tumor-and-Trauma-Mapping-with-3D-AR-Based-Visualization-for-Pre-Surgical-Planning.git
   ```
2. Navigate to the project directory:
   ```sh
   cd brats-viewer-ui
   ```
3. Install the dependencies:
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

### Running the Development Server

Run the following command to start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
