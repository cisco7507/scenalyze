import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  // Read VITE_* variables from the repo-root .env so dashboard deployments
  // can be configured from the same topology file as the backend.
  envDir: '..',
  plugins: [react(), tailwindcss()],
})
