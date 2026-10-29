/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: { extend: { fontFamily: { display: ["var(--font-display)"], body: ["var(--font-body)"] } } },
  plugins: [],
};
