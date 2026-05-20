/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Inter", "ui-sans-serif", "system-ui"],
      },
      boxShadow: {
        neon: "0 0 28px rgba(34, 211, 238, 0.22)",
        alert: "0 0 30px rgba(248, 113, 113, 0.28)",
      },
    },
  },
  plugins: [],
};
