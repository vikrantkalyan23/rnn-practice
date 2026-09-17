import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Next Word Predictor",
  description: "A small LSTM next-word prediction interface",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
