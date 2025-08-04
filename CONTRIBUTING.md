# Contributing to Breast Cancer Ultrasound Classification

Thank you for your interest in contributing to this project! We welcome contributions of all kinds.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- CUDA-compatible GPU (recommended for training)

### Development Setup

1. **Fork and clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-ultrasound.git
cd breast-cancer-ultrasound
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r webapp/requirements.txt
```

4. **Run tests to ensure everything works**
```bash
python webapp_validation_suite.py
```

## 🎯 Types of Contributions

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include detailed reproduction steps
- Provide system information (OS, Python version, etc.)
- Include error messages and stack traces

### 💡 Feature Requests
- Open a GitHub discussion first
- Describe the use case and benefits
- Consider implementation complexity
- Provide mockups or examples if applicable

### 🔧 Code Contributions
- Model improvements
- Web application enhancements
- Documentation updates
- Performance optimizations
- New visualization features

### 📚 Documentation
- README improvements
- Code comments
- Tutorial creation
- API documentation

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused

### Testing
- Write tests for new features
- Ensure all tests pass before submitting
- Include both unit tests and integration tests
- Test on multiple datasets if possible

### Model Development
- Document training procedures
- Include performance metrics
- Provide visualization of results
- Consider computational efficiency

### Web Application
- Ensure responsive design
- Test on multiple browsers
- Maintain accessibility standards
- Include error handling

## 🔄 Pull Request Process

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
- Follow the coding standards
- Add tests for new functionality
- Update documentation as needed

3. **Test your changes**
```bash
# Run the validation suite
python webapp_validation_suite.py

# Test specific components
python test_fixed_webapp.py
python src/test_fixed_model.py
```

4. **Commit your changes**
```bash
git add .
git commit -m "feat: add new feature description"
```

5. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**
- Use a clear title and description
- Reference any related issues
- Include screenshots for UI changes
- List any breaking changes

## 📝 Commit Message Format

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add GradCAM visualization to web app
fix: resolve preprocessing pipeline mismatch
docs: update installation instructions
test: add validation suite for model accuracy
```

## 🧪 Testing Requirements

### Before Submitting a PR

1. **Run all validation tests**
```bash
python webapp_validation_suite.py
```

2. **Test model performance**
```bash
python src/test_fixed_model.py
```

3. **Test web application**
```bash
streamlit run webapp/app.py
# Manually test file uploads and predictions
```

4. **Check code quality**
```bash
# If you have these tools installed
flake8 .
black --check .
```

### Test Coverage Areas
- Model loading and inference
- Image preprocessing pipeline
- Web application functionality
- GradCAM visualization
- Data validation
- Error handling

## 🚦 Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation** review
5. **Final approval** and merge

## 🎨 Adding New Features

### Model Improvements
- Compare against existing baseline (96.58% accuracy)
- Provide training scripts and configuration
- Include performance comparison tables
- Document computational requirements

### Web Application Features
- Maintain consistency with existing design
- Add appropriate error handling
- Include user feedback mechanisms
- Test with various image formats

### Visualization Features
- Ensure scalability for different image sizes
- Provide multiple visualization options
- Include save/export functionality
- Maintain performance standards

## 📊 Performance Standards

### Model Performance
- Accuracy should be ≥ 95% on test set
- Training time should be reasonable (< 2 hours)
- Model size should be < 100MB when possible
- Memory usage should be documented

### Web Application
- Page load time < 3 seconds
- Prediction time < 5 seconds
- Support for images up to 10MB
- Works on mobile devices

## 🤔 Need Help?

- **Documentation**: Check the README.md and MODEL_SUMMARY.md
- **Issues**: Browse existing GitHub issues
- **Discussions**: Start a GitHub discussion
- **Contact**: Reach out to project maintainers

## 📜 Code of Conduct

### Our Pledge
We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special mentions for first-time contributors
- Attribution in academic papers (if applicable)

Thank you for contributing to advancing breast cancer detection technology! 🙏
