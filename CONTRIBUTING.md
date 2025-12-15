# Contributing to AIDocumentIndexer

Thank you for your interest in contributing to AIDocumentIndexer! This document provides guidelines and information for contributors.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python/Node version, etc.)
   - Relevant logs or screenshots

### Suggesting Features

1. **Check existing feature requests**
2. **Open a discussion** or issue describing:
   - The problem you're trying to solve
   - Your proposed solution
   - Alternative solutions considered
   - Any potential drawbacks

### Contributing Code

#### Getting Started

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/AIDocumentIndexer.git
   cd AIDocumentIndexer
   ```
3. **Set up development environment:**
   ```bash
   # Backend
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Frontend
   cd ../frontend
   npm install
   ```
4. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

#### Making Changes

1. **Write clean, readable code**
2. **Follow existing code style**
3. **Add tests for new functionality**
4. **Update documentation as needed**
5. **Keep commits focused and atomic**

#### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(chat): add streaming support for chat responses
fix(upload): handle large files properly
docs(api): update authentication endpoint docs
```

#### Pull Request Process

1. **Ensure tests pass:**
   ```bash
   # Backend
   cd backend && pytest

   # Frontend
   cd frontend && npm test
   ```

2. **Run linting:**
   ```bash
   # Backend
   ruff check .
   mypy .

   # Frontend
   npm run lint
   ```

3. **Update documentation** if needed

4. **Create pull request:**
   - Use a clear title
   - Reference related issues
   - Describe your changes
   - Include screenshots for UI changes

5. **Address review feedback**

6. **Wait for approval** from maintainers

## Development Guidelines

### Python (Backend)

- Use Python 3.11+
- Follow PEP 8 style guide
- Use type hints
- Write docstrings for functions
- Use `structlog` for logging
- Prefer async functions for I/O

```python
async def process_document(
    document_id: str,
    options: ProcessingOptions,
) -> ProcessingResult:
    """
    Process a document with the given options.

    Args:
        document_id: Unique document identifier
        options: Processing configuration

    Returns:
        ProcessingResult containing status and metrics
    """
    logger.info("Processing document", document_id=document_id)
    ...
```

### TypeScript (Frontend)

- Use TypeScript strict mode
- Prefer functional components
- Use React Query for data fetching
- Follow component structure:

```typescript
// components/feature/FeatureComponent.tsx
interface FeatureProps {
  id: string;
  onAction: (id: string) => void;
}

export function FeatureComponent({ id, onAction }: FeatureProps) {
  const { data, isLoading } = useFeatureData(id);

  if (isLoading) return <Skeleton />;

  return (
    <div className="feature">
      {/* Component content */}
    </div>
  );
}
```

### Testing

**Backend Tests:**
- Use pytest
- Place tests in `backend/tests/`
- Name files `test_*.py`
- Use fixtures from `conftest.py`

```python
@pytest.mark.asyncio
async def test_process_document(db_session, mock_llm):
    result = await process_document("doc-123", options)
    assert result.status == "completed"
```

**Frontend Tests:**
- Use Jest and React Testing Library
- Place tests in `frontend/__tests__/`
- Test user interactions, not implementation

```typescript
describe('FeatureComponent', () => {
  it('displays data when loaded', async () => {
    render(<FeatureComponent id="123" />);
    expect(await screen.findByText('Feature Data')).toBeInTheDocument();
  });
});
```

### Documentation

- Update relevant docs when making changes
- Use clear, concise language
- Include code examples
- Keep README.md up to date

## Project Structure

```
AIDocumentIndexer/
├── backend/           # Python FastAPI backend
│   ├── api/           # API routes and middleware
│   ├── services/      # Business logic
│   ├── db/            # Database models
│   └── tests/         # Backend tests
├── frontend/          # Next.js frontend
│   ├── app/           # App Router pages
│   ├── components/    # React components
│   ├── lib/           # Utilities and API client
│   └── __tests__/     # Frontend tests
├── docker/            # Docker configurations
├── docs/              # Documentation
└── scripts/           # Utility scripts
```

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` - these are suitable for newcomers.

### Feature Ideas

- Additional LLM providers
- New document formats
- UI improvements
- Performance optimizations
- Documentation improvements
- Test coverage

### Documentation

- Improve existing docs
- Add tutorials
- Create how-to guides
- Translate documentation

## Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Pull Requests**: Code contributions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to AIDocumentIndexer!
