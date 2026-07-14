import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NIRGuard Secure Viewer",
  description:
    "Controlled review watermark validation and leak attribution console.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
