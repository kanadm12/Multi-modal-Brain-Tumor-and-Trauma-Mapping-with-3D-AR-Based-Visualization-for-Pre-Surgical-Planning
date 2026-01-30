// custom.d.ts
import { AriaAttributes, DOMAttributes } from 'react';

declare global {
  namespace React {
    interface HTMLAttributes<T> extends AriaAttributes, DOMAttributes<T> {
      webkitdirectory?: string;
    }
  }
}