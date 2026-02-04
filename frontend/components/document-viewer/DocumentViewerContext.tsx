"use client";

/**
 * Document Viewer Context
 * ========================
 *
 * Provides a global context for opening documents in full-screen mode
 * from anywhere in the application.
 *
 * Usage:
 *   const { openDocument, closeDocument } = useDocumentViewer();
 *   openDocument({ id: "...", url: "...", name: "doc.pdf", type: "pdf" });
 */

import React, { createContext, useContext, useState, useCallback } from "react";
import { FullScreenViewer } from "./FullScreenViewer";

// Types
interface DocumentInfo {
  id: string;
  url: string;
  name: string;
  type: string;
  initialPage?: number;
  documentData?: any; // Full document object for metadata sidebar
}

interface DocumentViewerContextType {
  isOpen: boolean;
  document: DocumentInfo | null;
  openDocument: (doc: DocumentInfo) => void;
  closeDocument: () => void;
}

// Context
const DocumentViewerContext = createContext<DocumentViewerContextType | null>(null);

// Provider
interface DocumentViewerProviderProps {
  children: React.ReactNode;
}

export function DocumentViewerProvider({ children }: DocumentViewerProviderProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [document, setDocument] = useState<DocumentInfo | null>(null);

  const openDocument = useCallback((doc: DocumentInfo) => {
    setDocument(doc);
    setIsOpen(true);
  }, []);

  const closeDocument = useCallback(() => {
    setIsOpen(false);
    // Delay clearing document to allow for exit animation
    setTimeout(() => setDocument(null), 300);
  }, []);

  return (
    <DocumentViewerContext.Provider
      value={{ isOpen, document, openDocument, closeDocument }}
    >
      {children}
      {isOpen && document && (
        <FullScreenViewer
          documentId={document.id}
          documentUrl={document.url}
          documentName={document.name}
          documentType={document.type}
          initialPage={document.initialPage}
          documentData={document.documentData}
          onClose={closeDocument}
        />
      )}
    </DocumentViewerContext.Provider>
  );
}

// Hook
export function useDocumentViewer() {
  const context = useContext(DocumentViewerContext);
  if (!context) {
    throw new Error("useDocumentViewer must be used within a DocumentViewerProvider");
  }
  return context;
}
