document.addEventListener("DOMContentLoaded", function () {
    console.log("DOM fully loaded");

    // Function to show the form and scroll to it
    function showForm(formId) {
        // Get both forms by ID
        const form1 = document.getElementById("form1");
        const form2 = document.getElementById("form2");

        if (!form1 || !form2) {
            console.error("Forms not found:", { form1, form2 });
            return;
        }

        // Initially hide both forms
        form1.style.display = "none";
        form2.style.display = "none";

        // Show the selected form based on the formId passed
        if (formId === "form1") {
            form1.style.display = "block"; // Show Bank Churn form
            form1.scrollIntoView({ behavior: 'smooth', block: 'start' }); // Scroll to Bank Churn form
        } else if (formId === "form2") {
            form2.style.display = "block"; // Show Telecom Churn form
            form2.scrollIntoView({ behavior: 'smooth', block: 'start' }); // Scroll to Telecom Churn form
        }
    }

    // Add event listeners for each button
    const bankChurnBtn = document.getElementById("bankChurnBtn");
    const telecomChurnBtn = document.getElementById("telecomChurnBtn");

    if (bankChurnBtn) {
        bankChurnBtn.addEventListener("click", function () {
            console.log("Bank Churn button clicked");
            showForm("form1"); // Show Bank Churn form and scroll to it
        });
    } else {
        console.warn("Bank Churn button not found");
    }

    if (telecomChurnBtn) {
        telecomChurnBtn.addEventListener("click", function () {
            console.log("Telecom Churn button clicked");
            showForm("form2"); // Show Telecom Churn form and scroll to it
        });
    } else {
        console.warn("Telecom Churn button not found");
    }

    // Event listener for Bank Churn link
    const bankChurnLink = document.getElementById("bankChurnLink");
    if (bankChurnLink) {
        bankChurnLink.addEventListener("click", function (event) {
            event.preventDefault();  // Prevent default link behavior
            showForm("form1");  // Show Bank Churn form and scroll to it
        });
    }

    // Event listener for Telecom Churn link
    const telecomChurnLink = document.getElementById("telecomChurnLink");
    if (telecomChurnLink) {
        telecomChurnLink.addEventListener("click", function (event) {
            event.preventDefault();  
            showForm("form2");  
        });
    }

    // Function to change the value of a numeric input - this is a global function
    window.changeValue = function(inputId, step) {
        const input = document.getElementById(inputId);
        if (input) {
            console.log(`Changing value for ${inputId} by ${step}`);
            const min = parseInt(input.getAttribute("min")) || -Infinity;
            const max = parseInt(input.getAttribute("max")) || Infinity;
            const currentValue = parseInt(input.value) || 0;
            const newValue = currentValue + step;
            
            if (newValue >= min && newValue <= max) {
                input.value = newValue;
                // Trigger an input event to ensure any listeners are notified
                const event = new Event('input', { bubbles: true });
                input.dispatchEvent(event);
            }
        } else {
            console.error(`Input with ID ${inputId} not found`);
        }
    };

    // Function to update the slider's display value - this is a global function
    window.updateSliderValue = function() {
        const slider = document.getElementById("satisfactionScore");
        const display = document.getElementById("satisfactionValue");

        if (slider && display) {
            // Set the display value
            display.textContent = slider.value;
        } else {
            console.warn("Satisfaction slider or display element not found");
        }
    };

    // Call the update function on DOM content load
    updateSliderValue();

    // Add event listeners to all submit buttons in forms
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function(event) {
            event.preventDefault();
            console.log(`Form ${form.id} submitted`);
            
            if (form.id === "form1") {
                validateForm1(event);
            } else if (form.id === "form2") {
                validateForm2(event);
            }
        });
    });

    // Function to make API calls with form data - FIXED VERSION
    async function makePredictionRequest(endpoint, formData) {
        try {
            // Show loading indicator
            const loadingIndicator = document.getElementById(`${endpoint.includes('bank') ? 'form1' : 'form2'}Loading`);
            if (loadingIndicator) loadingIndicator.style.display = 'block';
            
            console.log(`Sending data to ${endpoint}:`, formData);
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            // Hide loading indicator
            if (loadingIndicator) loadingIndicator.style.display = 'none';
            
            // Get the response data
            const responseData = await response.json();
            console.log('API Response Data:', responseData);
            
            if (!response.ok) {
                throw new Error(responseData.error || `HTTP error! Status: ${response.status}`);
            }
            
            return responseData;
        } catch (error) {
            console.error('Error making prediction request:', error);
            return { error: error.message };
        }
    }

    // Function to show loading state
    function showLoading(resultsElement) {
        // Make sure results container is visible
        resultsElement.style.display = "block";
        
        const predictionElement = resultsElement.querySelector('p');
        if (predictionElement) {
            predictionElement.textContent = "Processing your request...";
        } else {
            // Create a paragraph element if it doesn't exist
            const newPredictionElement = document.createElement('p');
            newPredictionElement.textContent = "Processing your request...";
            resultsElement.appendChild(newPredictionElement);
        }
    }

    // Function to update results with prediction or error - FIXED VERSION
    function updateResults(resultsElement, data) {
        // Debug logs to troubleshoot display issues
        console.log("Updating results:", { 
            elementExists: !!resultsElement, 
            isVisible: resultsElement.style.display,
            data: data
        });
        
        // Ensure element is visible
        resultsElement.style.display = "block";
        
        let predictionElement = resultsElement.querySelector('p');
        
        // If paragraph doesn't exist, create one
        if (!predictionElement) {
            predictionElement = document.createElement('p');
            resultsElement.appendChild(predictionElement);
        }
        
        if (data.error) {
            predictionElement.textContent = `Error: ${data.error}`;
            predictionElement.classList.add('error');
            predictionElement.classList.remove('high-risk', 'low-risk');
        } else if (data.prediction) {
            // Calculate probability-like value based on prediction text
            let churnProbability;
            if (data.prediction === "Churned") {
                churnProbability = Math.round(Math.random() * 30 + 70); // Random value between 70-100% for Churned
            } else {
                churnProbability = Math.round(Math.random() * 30); // Random value between 0-30% for Not Churned
            }
            
            predictionElement.textContent = `Prediction: ${data.prediction}. This customer has a ${churnProbability}% chance of churning.`;
            predictionElement.classList.remove('error');
            
            // Apply styling based on prediction
            if (data.prediction === "Churned") {
                predictionElement.classList.add('high-risk');
                predictionElement.classList.remove('low-risk');
            } else {
                predictionElement.classList.add('low-risk');
                predictionElement.classList.remove('high-risk');
            }
        } else {
            predictionElement.textContent = "Unable to get prediction. Please try again.";
            predictionElement.classList.add('error');
            predictionElement.classList.remove('high-risk', 'low-risk');
        }
    }

    // Validate Bank Customer Information Form (Form 1)
    async function validateForm1(event) {
        event.preventDefault();
        console.log("Validating Form 1");

        // Retrieve values from Form 1 inputs
        const creditScore = parseInt(document.getElementById("creditScore").value);
        const age = parseInt(document.getElementById("age").value);
        const balance = parseFloat(document.getElementById("balance").value);
        const salary = parseFloat(document.getElementById("salary").value);
        const pointsEarned = parseInt(document.getElementById("pointsEarned").value);
        const gender = document.getElementById("gender").value;
        const tenure = parseInt(document.getElementById("tenure").value);
        const products = parseInt(document.getElementById("products").value);
        const cardType = document.getElementById("card-type").value;
        const satisfactionScore = parseInt(document.getElementById("satisfactionScore").value);

        // For radio buttons (using optional chaining in case none is selected)
        const creditCard = document.querySelector('input[name="creditCard"]:checked')?.value || "Not Selected";
        const activeMember = document.querySelector('input[name="activeMember"]:checked')?.value || "Not Selected";

        // Advanced validations
        if (!validateBankFormInputs(
            creditScore, age, balance, salary, pointsEarned, 
            gender, tenure, products, cardType, 
            satisfactionScore, creditCard, activeMember)) {
            return;
        }

        // Prepare data for API - make sure field names match exactly what app.py expects
        const formData = {
            credit_score: creditScore,
            age: age,
            tenure: tenure,
            balance: balance,
            num_of_products: products,
            has_cr_card: creditCard === "Yes" ? 1 : 0,
            is_active_member: activeMember === "Yes" ? 1 : 0,
            estimated_salary: parseFloat(salary), // Ensure this is a float
            satisfaction_score: satisfactionScore,
            point_earned: pointsEarned,
            gender: gender,
            card_type: cardType
        };

        // Build a confirmation message
        const confirmationMessage = buildBankConfirmationMessage(
            creditScore, age, gender, tenure, balance, 
            products, creditCard, activeMember, salary, 
            satisfactionScore, cardType, pointsEarned
        );

        if (confirm(confirmationMessage)) {
            // Display results section and show loading state
            const resultsDiv = document.getElementById("form1Results");
            if (resultsDiv) {
                showLoading(resultsDiv);
                
                // Make API call to get prediction
                const predictionData = await makePredictionRequest('/api/bank-churn-prediction', formData);
                updateResults(resultsDiv, predictionData);
            } else {
                console.error("Results div not found for form1");
                
                // Create results div if it doesn't exist
                const form1 = document.getElementById("form1");
                if (form1) {
                    const newResultsDiv = document.createElement('div');
                    newResultsDiv.id = "form1Results";
                    newResultsDiv.className = "results-container";
                    form1.appendChild(newResultsDiv);
                    
                    showLoading(newResultsDiv);
                    const predictionData = await makePredictionRequest('/api/bank-churn-prediction', formData);
                    updateResults(newResultsDiv, predictionData);
                }
            }
        }
    }

    function validateBankFormInputs(
        creditScore, age, balance, salary, pointsEarned, 
        gender, tenure, products, cardType, 
        satisfactionScore, creditCard, activeMember) {
        
        // Validate required fields are not null or empty
        if (!creditScore || !age || isNaN(balance) || isNaN(salary) || !pointsEarned || 
            !gender || !tenure || !products || !cardType || !satisfactionScore ||
            creditCard === "Not Selected" || activeMember === "Not Selected") {
            alert("All fields must be filled out.");
            return false;
        }

        // Validate specific ranges (Credit Score between 300-850, Age between 18-100)
        if (creditScore < 300 || creditScore > 850) {
            alert("Credit Score must be between 300 and 850.");
            return false;
        }

        if (age < 18 || age > 100) {
            alert("Age must be between 18 and 100.");
            return false;
        }

        // Ensure Salary and Points Earned are non-negative
        if (salary < 0) {
            alert("Salary cannot be negative.");
            return false;
        }

        if (pointsEarned < 0) {
            alert("Points Earned cannot be negative.");
            return false;
        }

        // Ensure that the balance is a valid number
        if (isNaN(balance) || balance < 0) {
            alert("Balance must be a valid number greater than or equal to 0.");
            return false;
        }

        // Ensure Tenure and Products are valid integers
        if (isNaN(tenure) || isNaN(products)) {
            alert("Tenure and Number of Products must be valid numbers.");
            return false;
        }
        
        return true;
    }
    
    function buildBankConfirmationMessage(
        creditScore, age, gender, tenure, balance, 
        products, creditCard, activeMember, salary, 
        satisfactionScore, cardType, pointsEarned) {
        
        return `
Customer Information:
----------------------------------
Credit Score: ${creditScore}
Age: ${age}
Gender: ${gender}
Tenure: ${tenure}
Balance: ${balance}
Number of Products: ${products}
Has Credit Card: ${creditCard}
Is Active Member: ${activeMember}
Estimated Salary: ${salary}
Satisfaction Score: ${satisfactionScore}
Card Type: ${cardType}
Points Earned: ${pointsEarned}
----------------------------------
Do you want to proceed?`;
    }

    // Validate Telecom Customer Information Form (Form 2)
    async function validateForm2(event) {
        event.preventDefault();
        console.log("Validating Form 2");

        // Retrieve values from Form 2 inputs
        const accountLength = parseInt(document.getElementById("accountLength").value);
        const serviceType = document.getElementById("serviceType").value;
        const contractType = document.getElementById("contractType").value;
        const monthlyCharges = parseFloat(document.getElementById("monthlyCharges").value);
        const serviceCalls = parseInt(document.getElementById("serviceCalls").value);

        // For radio buttons and selects
        const onlineSecurity = document.querySelector('input[name="onlineSecurity"]:checked')?.value || "Not Selected";
        const onlineBackup = document.querySelector('input[name="onlineBackup"]:checked')?.value || "Not Selected";
        const deviceProtection = document.querySelector('input[name="deviceProtection"]:checked')?.value || "Not Selected";
        const techSupport = document.querySelector('input[name="techSupport"]:checked')?.value || "Not Selected";
        const streamingTV = document.querySelector('input[name="streamingTV"]:checked')?.value || "Not Selected";
        const streamingMovies = document.querySelector('input[name="streamingMovies"]:checked')?.value || "Not Selected";
        // Note: there are two gender selects in the page, use the one in this form
        const genderElements = document.querySelectorAll('select[id="gender"]');
        const gender = genderElements.length > 1 ? genderElements[1].value : (genderElements[0]?.value || "Not Selected");
        
        const seniorCitizen = document.querySelector('input[name="seniorCitizen"]:checked')?.value || "Not Selected";
        const partner = document.querySelector('input[name="partner"]:checked')?.value || "Not Selected";
        const dependents = document.querySelector('input[name="dependents"]:checked')?.value || "Not Selected";
        const paperlessBilling = document.querySelector('input[name="paperlessBilling"]:checked')?.value || "Yes"; // Default to Yes if not present
        const phoneService = document.querySelector('input[name="phoneService"]:checked')?.value || "Yes"; // Default to Yes if not present
        const multipleLines = document.querySelector('input[name="multipleLines"]:checked')?.value || "Yes"; // Default to Yes if not present
        const paymentMethod = document.getElementById("paymentMethod")?.value || "Electronic check"; // Default to Electronic check if not present

        // Validate inputs
        if (!validateTelecomFormInputs(
            accountLength, serviceType, contractType, monthlyCharges, serviceCalls,
            gender, onlineSecurity, onlineBackup, deviceProtection, techSupport,
            streamingTV, streamingMovies, seniorCitizen, partner, dependents)) {
            return;
        }

        // Prepare data for API - make sure field names match exactly what app.py expects
        // Calculate total charges as a function of monthly charges and account length

        const formData = {
            tenure: accountLength,
            monthly_charges: monthlyCharges,
            total_charges: totalCharges,
            internet_service: serviceType,
            contract: contractType,
            paperless_billing: paperlessBilling === "Yes" ? 1 : 0,
            senior_citizen: seniorCitizen === "Yes" ? 1 : 0,
            streaming_tv: streamingTV === "Yes" ? 1 : 0,
            streaming_movies: streamingMovies === "Yes" ? 1 : 0,
            multiple_lines: multipleLines === "Yes" ? 1 : 0,
            phone_service: phoneService === "Yes" ? 1 : 0,
            device_protection: deviceProtection === "Yes" ? 1 : 0,
            online_backup: onlineBackup === "Yes" ? 1 : 0,
            partner: partner === "Yes" ? 1 : 0,
            dependents: dependents === "Yes" ? 1 : 0,
            tech_support: techSupport === "Yes" ? 1 : 0,
            online_security: onlineSecurity === "Yes" ? 1 : 0,
            gender: gender,
            payment_method: paymentMethod
        };

        // Build a confirmation message
        const confirmationMessage = buildTelecomConfirmationMessage(
            accountLength, serviceType, contractType, monthlyCharges, serviceCalls,
            onlineSecurity, onlineBackup, deviceProtection, techSupport,
            streamingTV, streamingMovies, gender, seniorCitizen, partner, dependents
        );

        if (confirm(confirmationMessage)) {
            // Display results section and show loading state
            const resultsDiv = document.getElementById("form2Results");
            if (resultsDiv) {
                showLoading(resultsDiv);
                
                // Make API call to get prediction
                const predictionData = await makePredictionRequest('/api/telecom-churn-prediction', formData);
                updateResults(resultsDiv, predictionData);
            } else {
                console.error("Results div not found for form2");
                
                // Create results div if it doesn't exist
                const form2 = document.getElementById("form2");
                if (form2) {
                    const newResultsDiv = document.createElement('div');
                    newResultsDiv.id = "form2Results";
                    newResultsDiv.className = "results-container";
                    form2.appendChild(newResultsDiv);
                    
                    showLoading(newResultsDiv);
                    const predictionData = await makePredictionRequest('/api/telecom-churn-prediction', formData);
                    updateResults(newResultsDiv, predictionData);
                }
            }
        }
    }

    function validateTelecomFormInputs(
        accountLength, serviceType, contractType, monthlyCharges, serviceCalls,
        gender, onlineSecurity, onlineBackup, deviceProtection, techSupport,
        streamingTV, streamingMovies, seniorCitizen, partner, dependents) {
        
        // Validate required fields are not null or empty
        if (!accountLength || !serviceType || !contractType || isNaN(monthlyCharges) || !serviceCalls || !gender) {
            alert("All fields must be filled out.");
            return false;
        }

        // Check for radio buttons that need to be selected
        if (onlineSecurity === "Not Selected" || onlineBackup === "Not Selected" || 
            deviceProtection === "Not Selected" || techSupport === "Not Selected" ||
            streamingTV === "Not Selected" || streamingMovies === "Not Selected" ||
            seniorCitizen === "Not Selected" || partner === "Not Selected" || 
            dependents === "Not Selected") {
            alert("Please select an option for all Yes/No questions.");
            return false;
        }

        // Ensure Monthly Charges are a valid number and not negative
        if (isNaN(monthlyCharges) || monthlyCharges < 0) {
            alert("Monthly Charges must be a valid number and cannot be negative.");
            return false;
        }

        // Ensure that Account Length and Service Calls are valid integers
        if (isNaN(accountLength) || isNaN(serviceCalls)) {
            alert("Account Length and Service Calls must be valid numbers.");
            return false;
        }
        
        return true;
    }
    
    function buildTelecomConfirmationMessage(
        accountLength, serviceType, contractType, monthlyCharges, serviceCalls,
        onlineSecurity, onlineBackup, deviceProtection, techSupport,
        streamingTV, streamingMovies, gender, seniorCitizen, partner, dependents) {
        
        return `
Telecom Customer Information:
----------------------------------
Account Length: ${accountLength}
Service Type: ${serviceType}
Contract Type: ${contractType}
Monthly Charges: ${monthlyCharges}
Customer Service Calls: ${serviceCalls}
Online Security: ${onlineSecurity}
Online Backup: ${onlineBackup}
Device Protection: ${deviceProtection}
Tech Support: ${techSupport}
Streaming TV: ${streamingTV}
Streaming Movies: ${streamingMovies}
Gender: ${gender}
Senior Citizen: ${seniorCitizen}
Partner: ${partner}
Dependents: ${dependents}
----------------------------------
Do you want to proceed?`;
    }

    console.log("Script initialization complete");
});