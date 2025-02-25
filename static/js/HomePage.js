document.addEventListener("DOMContentLoaded", function () {
    console.log("DOM fully loaded"); // Add this for debugging

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

    // Function to change the value of a numeric input
    function changeValue(inputId, step) {
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
    }

    // Attach event listeners for the + and - buttons
    const plusBtns = document.querySelectorAll(".plus");
    const minusBtns = document.querySelectorAll(".minus");

    console.log(`Found ${plusBtns.length} plus buttons and ${minusBtns.length} minus buttons`);

    plusBtns.forEach((btn) => {
        btn.addEventListener("click", function (e) {
            e.preventDefault(); // Prevent default button behavior
            const inputId = btn.getAttribute("data-input-id");
            console.log(`Plus button clicked for ${inputId}`);
            changeValue(inputId, 1);
        });
    });

    minusBtns.forEach((btn) => {
        btn.addEventListener("click", function (e) {
            e.preventDefault(); // Prevent default button behavior
            const inputId = btn.getAttribute("data-input-id");
            console.log(`Minus button clicked for ${inputId}`);
            changeValue(inputId, -1);
        });
    });

    // Function to update the slider's display value
    function updateSliderValue() {
        const slider = document.getElementById("satisfactionScore");
        const display = document.getElementById("satisfactionValue");

        if (slider && display) {
            // Set the display value initially
            display.textContent = slider.value;

            // Update the display value whenever the slider value changes
            slider.addEventListener("input", function () {
                display.textContent = slider.value;
            });
        } else {
            console.warn("Satisfaction slider or display element not found");
        }
    }

    // Call the update function on DOM content load
    updateSliderValue();

    // --- Form Validation Functions ---

    // Validate Bank Customer Information Form (Form 1)
    function validateForm1(event) {
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
        const satisfactionScore = document.getElementById("satisfactionScore").value;

        console.log("Form 1 values:", { creditScore, age, balance, salary, pointsEarned, gender, tenure, products, cardType, satisfactionScore });

        // For radio buttons (using optional chaining in case none is selected)
        const creditCard = document.querySelector('input[name="creditCard"]:checked')?.value || "Not Selected";
        const activeMember = document.querySelector('input[name="activeMember"]:checked')?.value || "Not Selected";

        // Advanced validations

        // Validate required fields are not null or empty
        if (!creditScore || !age || !balance || !salary || !pointsEarned || !gender || !tenure || !products || !cardType || !satisfactionScore) {
            alert("All fields must be filled out.");
            return;
        }

        // Validate specific ranges (Credit Score between 300-850, Age between 18-100)
        if (creditScore < 300 || creditScore > 850) {
            alert("Credit Score must be between 300 and 850.");
            return;
        }

        if (age < 18 || age > 100) {
            alert("Age must be between 18 and 100.");
            return;
        }

        // Ensure Salary and Points Earned are non-negative
        if (salary < 0) {
            alert("Salary cannot be negative.");
            return;
        }

        if (pointsEarned < 0) {
            alert("Points Earned cannot be negative.");
            return;
        }

        // Ensure that the balance is a valid number
        if (isNaN(balance) || balance < 0) {
            alert("Balance must be a valid number greater than or equal to 0.");
            return;
        }

        // Ensure Tenure and Products are valid integers
        if (isNaN(tenure) || isNaN(products)) {
            alert("Tenure and Number of Products must be valid numbers.");
            return;
        }

        // Build a confirmation message
        const confirmationMessage = `
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

        if (confirm(confirmationMessage)) {
            alert("Form 1 submitted successfully!");
            // Get the form element
            const form = document.getElementById("form1");
            if (form) {
                form.submit();
            } else {
                console.error("Form element not found");
                alert("Error: Form element not found");
            }
        }
    }

    // Validate Telecom Customer Information Form (Form 2)
    function validateForm2(event) {
        event.preventDefault();
        console.log("Validating Form 2");

        // Retrieve values from Form 2 inputs
        const accountLength = parseInt(document.getElementById("accountLength").value);
        const serviceType = document.getElementById("serviceType").value;
        const contractType = document.getElementById("contractType").value;
        const monthlyCharges = parseFloat(document.getElementById("monthlyCharges").value);
        const serviceCalls = parseInt(document.getElementById("serviceCalls").value);

        console.log("Form 2 values:", { accountLength, serviceType, contractType, monthlyCharges, serviceCalls });

        // For radio buttons
        const onlineSecurity = document.querySelector('input[name="onlineSecurity"]:checked')?.value || "Not Selected";
        const onlineBackup = document.querySelector('input[name="onlineBackup"]:checked')?.value || "Not Selected";
        const deviceProtection = document.querySelector('input[name="deviceProtection"]:checked')?.value || "Not Selected";
        const techSupport = document.querySelector('input[name="techSupport"]:checked')?.value || "Not Selected";
        const streamingTV = document.querySelector('input[name="streamingTV"]:checked')?.value || "Not Selected";
        const streamingMovies = document.querySelector('input[name="streamingMovies"]:checked')?.value || "Not Selected";
        const gender = document.querySelector('select[name="gender"]')?.value || document.getElementById("gender").value;
        const seniorCitizen = document.querySelector('input[name="seniorCitizen"]:checked')?.value || "Not Selected";
        const partner = document.querySelector('input[name="partner"]:checked')?.value || "Not Selected";
        const dependents = document.querySelector('input[name="dependents"]:checked')?.value || "Not Selected";

        // Advanced validations

        // Validate required fields are not null or empty
        if (!accountLength || !serviceType || !contractType || !monthlyCharges || !serviceCalls || !gender) {
            alert("All fields must be filled out.");
            return;
        }

        // Ensure Monthly Charges are a valid number and not negative
        if (isNaN(monthlyCharges) || monthlyCharges < 0) {
            alert("Monthly Charges must be a valid number and cannot be negative.");
            return;
        }

        // Ensure that Account Length and Service Calls are valid integers
        if (isNaN(accountLength) || isNaN(serviceCalls)) {
            alert("Account Length and Service Calls must be valid numbers.");
            return;
        }

        // Build a confirmation message
        const confirmationMessage = `
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

        if (confirm(confirmationMessage)) {
            alert("Form 2 submitted successfully!");
            // Get the form element
            const form = document.getElementById("form2");
            if (form) {
                form.submit();
            } else {
                console.error("Form element not found");
                alert("Error: Form element not found");
            }
        }
    }

    // Attach event listeners to validation functions
    const submitBtn1 = document.getElementById("submitBtn1");
    if (submitBtn1) {
        console.log("Adding event listener to submitBtn1");
        submitBtn1.addEventListener("click", validateForm1);
    } else {
        console.warn("Submit button 1 not found");
    }

    const submitBtn2 = document.getElementById("submitBtn2");
    if (submitBtn2) {
        console.log("Adding event listener to submitBtn2");
        submitBtn2.addEventListener("click", validateForm2);
    } else {
        console.warn("Submit button 2 not found");
    }

    console.log("Script initialization complete");
});