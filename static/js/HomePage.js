document.addEventListener("DOMContentLoaded", function () {
    // Function to display validation messages
    function showValidationMessage(inputElement, message, isError = true) {
        const existingMessage = inputElement.parentElement.querySelector('.validation-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `validation-message ${isError ? 'text-red-500' : 'text-green-500'} text-sm mt-1`;
        messageDiv.textContent = message;
        inputElement.parentElement.appendChild(messageDiv);
    }

    // Function to handle numeric input buttons (+ and -)
    window.handleNumericInput = function(inputId, isIncrement) {
        const input = document.getElementById(inputId);
        if (!input) return;

        let currentValue = parseFloat(input.value) || 0;
        let step = parseFloat(input.getAttribute('step')) || 1;
        let min = parseFloat(input.getAttribute('min'));
        let max = parseFloat(input.getAttribute('max'));

        let newValue = isIncrement ? currentValue + step : currentValue - step;

        if (!isNaN(min) && newValue < min) newValue = min;
        if (!isNaN(max) && newValue > max) newValue = max;

        input.value = newValue;
        validateField(input);
    };

    // Function to handle radio buttons and dropdowns
    window.changeValue = function(inputId, value) {
        const input = document.getElementById(inputId);
        if (!input) return;
        
        input.value = value;
        validateField(input);
    };

    // Function to validate a single field
    function validateField(input) {
        const value = input.value.trim();
        const type = input.type;
        const required = input.hasAttribute('required');
        const min = parseFloat(input.getAttribute('min'));
        const max = parseFloat(input.getAttribute('max'));

        // Clear existing validation message
        const existingMessage = input.parentElement.querySelector('.validation-message');
        if (existingMessage) {
            existingMessage.remove();
        }

        if (required && !value) {
            showValidationMessage(input, 'This field is required');
            return false;
        }

        if (type === 'number' && value) {
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                showValidationMessage(input, 'Please enter a valid number');
                return false;
            }
            if (!isNaN(min) && numValue < min) {
                showValidationMessage(input, `Value must be at least ${min}`);
                return false;
            }
            if (!isNaN(max) && numValue > max) {
                showValidationMessage(input, `Value must be no more than ${max}`);
                return false;
            }
        }

        // If we get here, the field is valid
        showValidationMessage(input, 'Valid', false);
        return true;
    }

    // Function to validate entire form
    function validateForm(formId) {
        const form = document.getElementById(formId);
        if (!form) {
            console.error(`Form ${formId} not found`);
            return false;
        }

        let isValid = true;
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            if (!validateField(input)) {
                isValid = false;
            }
        });

        return isValid;
    }

    // Function to prepare form data for submission
    function prepareFormData(formId) {
        const form = document.getElementById(formId);
        const formData = new FormData(form);
        const formObject = {};

        formData.forEach((value, key) => {
            const input = form.querySelector(`[name="${key}"]`);
            
            // Handle different input types
            if (input.type === 'number') {
                formObject[key] = parseFloat(value) || 0;
            } else if (input.type === 'checkbox') {
                formObject[key] = input.checked ? 1 : 0;
            } else if (input.type === 'radio') {
                formObject[key] = value;
            } else {
                formObject[key] = value;
            }
        });

        return formObject;
    }

    // Enhanced form submission function
    async function validateAndSubmitForm(formId, endpoint) {
        try {
            console.log(`Validating form ${formId}`);
            
            if (!validateForm(formId)) {
                console.log('Form validation failed');
                return;
            }

            const formData = prepareFormData(formId);
            console.log('Submitting form data:', formData);

            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const result = await response.json();
            
            // Show result
            const resultDiv = document.createElement('div');
            resultDiv.className = 'mt-4 p-4 rounded-lg ' + 
                (result.prediction === 'Churned' ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700');
            resultDiv.textContent = `Prediction: ${result.prediction}`;
            
            const form = document.getElementById(formId);
            const existingResult = form.querySelector('.prediction-result');
            if (existingResult) {
                existingResult.remove();
            }
            resultDiv.className += ' prediction-result';
            form.appendChild(resultDiv);

        } catch (error) {
            alert(error.message);
            console.error('Submission error:', error);
        }
    }

    // Function to show/hide forms
    function showForm(formId) {
        const forms = document.querySelectorAll('#form1, #form2');
        forms.forEach(form => form.style.display = 'none');

        const selectedForm = document.getElementById(formId);
        if (selectedForm) {
            selectedForm.style.display = 'block';
            selectedForm.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // Initialize form event listeners
    const forms = document.querySelectorAll('#form1, #form2');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.addEventListener('blur', () => validateField(input));
            input.addEventListener('input', () => validateField(input));
        });
    });

    // Form submission handlers
    ['form1', 'form2'].forEach(formId => {
        const form = document.getElementById(formId);
        const submitBtn = document.getElementById(`submitBtn${formId.slice(-1)}`);
        const endpoint = formId === 'form1' ? '/api/bank-churn-prediction' : '/api/telecom-churn-prediction';

        if (form) {
            form.addEventListener("submit", (event) => {
                event.preventDefault();
                validateAndSubmitForm(formId, endpoint);
            });
        }

        if (submitBtn) {
            submitBtn.addEventListener("click", (event) => {
                event.preventDefault();
                validateAndSubmitForm(formId, endpoint);
            });
        }
    });

    // Initialize navigation button listeners
    ['bank', 'telecom'].forEach(type => {
        const btn = document.getElementById(`${type}ChurnBtn`);
        const link = document.getElementById(`${type}ChurnLink`);
        const formId = type === 'bank' ? 'form1' : 'form2';

        if (btn) {
            btn.addEventListener("click", () => showForm(formId));
        }

        if (link) {
            link.addEventListener("click", (event) => {
                event.preventDefault();
                showForm(formId);
            });
        }
    });
});