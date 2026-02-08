// custom.d.ts - Extend React's InputHTMLAttributes to include webkitdirectory
import 'react';

declare module 'react' {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface InputHTMLAttributes<T> {
    webkitdirectory?: string;
    directory?: string;
  }
}
