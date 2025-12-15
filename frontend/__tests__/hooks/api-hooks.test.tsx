/**
 * AIDocumentIndexer - API Hooks Tests
 * ====================================
 *
 * Tests for React Query hooks.
 */

import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { queryKeys } from '@/lib/api/hooks';

// Create a wrapper for React Query
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
};

describe('queryKeys', () => {
  describe('documents', () => {
    it('generates all documents key', () => {
      expect(queryKeys.documents.all).toEqual(['documents']);
    });

    it('generates list key without params', () => {
      expect(queryKeys.documents.list()).toEqual(['documents', 'list', undefined]);
    });

    it('generates list key with params', () => {
      const params = { page: 1, page_size: 20 };
      expect(queryKeys.documents.list(params)).toEqual([
        'documents',
        'list',
        params,
      ]);
    });

    it('generates detail key', () => {
      expect(queryKeys.documents.detail('doc-123')).toEqual([
        'documents',
        'detail',
        'doc-123',
      ]);
    });

    it('generates search key', () => {
      expect(queryKeys.documents.search('test query')).toEqual([
        'documents',
        'search',
        'test query',
      ]);
    });

    it('generates collections key', () => {
      expect(queryKeys.documents.collections).toEqual([
        'documents',
        'collections',
      ]);
    });
  });

  describe('chat', () => {
    it('generates sessions key', () => {
      expect(queryKeys.chat.sessions).toEqual(['chat', 'sessions']);
    });

    it('generates session key', () => {
      expect(queryKeys.chat.session('session-123')).toEqual([
        'chat',
        'session',
        'session-123',
      ]);
    });
  });

  describe('upload', () => {
    it('generates status key', () => {
      expect(queryKeys.upload.status('file-123')).toEqual([
        'upload',
        'status',
        'file-123',
      ]);
    });

    it('generates queue key', () => {
      expect(queryKeys.upload.queue).toEqual(['upload', 'queue']);
    });

    it('generates supportedTypes key', () => {
      expect(queryKeys.upload.supportedTypes).toEqual([
        'upload',
        'supportedTypes',
      ]);
    });
  });

  describe('generation', () => {
    it('generates all key', () => {
      expect(queryKeys.generation.all).toEqual(['generation']);
    });

    it('generates jobs key without status', () => {
      expect(queryKeys.generation.jobs()).toEqual([
        'generation',
        'jobs',
        undefined,
      ]);
    });

    it('generates jobs key with status', () => {
      expect(queryKeys.generation.jobs('completed')).toEqual([
        'generation',
        'jobs',
        'completed',
      ]);
    });

    it('generates job key', () => {
      expect(queryKeys.generation.job('job-123')).toEqual([
        'generation',
        'job',
        'job-123',
      ]);
    });

    it('generates formats key', () => {
      expect(queryKeys.generation.formats).toEqual(['generation', 'formats']);
    });
  });

  describe('collaboration', () => {
    it('generates all key', () => {
      expect(queryKeys.collaboration.all).toEqual(['collaboration']);
    });

    it('generates sessions key', () => {
      expect(queryKeys.collaboration.sessions()).toEqual([
        'collaboration',
        'sessions',
        undefined,
      ]);
    });

    it('generates session key', () => {
      expect(queryKeys.collaboration.session('session-123')).toEqual([
        'collaboration',
        'session',
        'session-123',
      ]);
    });

    it('generates critiques key', () => {
      expect(queryKeys.collaboration.critiques('session-123')).toEqual([
        'collaboration',
        'critiques',
        'session-123',
      ]);
    });

    it('generates modes key', () => {
      expect(queryKeys.collaboration.modes).toEqual([
        'collaboration',
        'modes',
      ]);
    });
  });

  describe('scraper', () => {
    it('generates all key', () => {
      expect(queryKeys.scraper.all).toEqual(['scraper']);
    });

    it('generates jobs key', () => {
      expect(queryKeys.scraper.jobs('running')).toEqual([
        'scraper',
        'jobs',
        'running',
      ]);
    });

    it('generates job key', () => {
      expect(queryKeys.scraper.job('job-123')).toEqual([
        'scraper',
        'job',
        'job-123',
      ]);
    });
  });

  describe('costs', () => {
    it('generates all key', () => {
      expect(queryKeys.costs.all).toEqual(['costs']);
    });

    it('generates usage key', () => {
      expect(queryKeys.costs.usage('month')).toEqual([
        'costs',
        'usage',
        'month',
      ]);
    });

    it('generates history key', () => {
      expect(queryKeys.costs.history).toEqual(['costs', 'history']);
    });

    it('generates current key', () => {
      expect(queryKeys.costs.current('week')).toEqual([
        'costs',
        'current',
        'week',
      ]);
    });

    it('generates dashboard key', () => {
      expect(queryKeys.costs.dashboard).toEqual(['costs', 'dashboard']);
    });

    it('generates alerts key', () => {
      expect(queryKeys.costs.alerts).toEqual(['costs', 'alerts']);
    });

    it('generates pricing key', () => {
      expect(queryKeys.costs.pricing).toEqual(['costs', 'pricing']);
    });
  });

  describe('health', () => {
    it('generates health key', () => {
      expect(queryKeys.health).toEqual(['health']);
    });
  });
});

describe('Query key uniqueness', () => {
  it('different resources have unique keys', () => {
    const allKeys = [
      queryKeys.documents.all,
      queryKeys.chat.sessions,
      queryKeys.upload.queue,
      queryKeys.generation.all,
      queryKeys.collaboration.all,
      queryKeys.scraper.all,
      queryKeys.costs.all,
      queryKeys.health,
    ];

    // Convert to strings for comparison
    const keyStrings = allKeys.map((key) => JSON.stringify(key));
    const uniqueKeys = new Set(keyStrings);

    expect(uniqueKeys.size).toBe(allKeys.length);
  });

  it('parameterized keys are unique for different params', () => {
    const key1 = queryKeys.documents.list({ page: 1 });
    const key2 = queryKeys.documents.list({ page: 2 });

    expect(JSON.stringify(key1)).not.toBe(JSON.stringify(key2));
  });

  it('same params produce same keys', () => {
    const key1 = queryKeys.documents.list({ page: 1 });
    const key2 = queryKeys.documents.list({ page: 1 });

    expect(JSON.stringify(key1)).toBe(JSON.stringify(key2));
  });
});
