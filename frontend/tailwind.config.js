/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        slateBrand: "#102a43",
        sand: "#f7f3eb",
        accent: "#ef8354",
        moss: "#3c6e71",
      },
      boxShadow: {
        panel: "0 18px 45px rgba(16, 42, 67, 0.08)",
      },
    },
  },
  plugins: [],
};
