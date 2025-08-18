// Tab persistence for AI Recommendations
document.addEventListener('DOMContentLoaded', function() {
    // Get URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const activeTab = urlParams.get('tab');
    
    // Handle tab persistence after form submission
    const form = document.querySelector('form[action*="index"]');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Add tab parameter to form action
            const tabParam = document.createElement('input');
            tabParam.type = 'hidden';
            tabParam.name = 'tab';
            tabParam.value = 'ai-recommendations';
            this.appendChild(tabParam);
        });
    }
    
    // Set active tab based on URL parameter
    if (activeTab === 'ai-recommendations') {
        // Activate AI Recommendations tab
        const aiTab = document.getElementById('ai-tab');
        const aiContent = document.getElementById('ai-recommendations');
        const chatbotTab = document.getElementById('chatbot-tab');
        const chatbotContent = document.getElementById('chatbot-recommendations');
        
        if (aiTab && aiContent) {
            // Remove active classes from chatbot
            if (chatbotTab) chatbotTab.classList.remove('active');
            if (chatbotContent) {
                chatbotContent.classList.remove('show', 'active');
                chatbotContent.classList.add('fade');
            }
            
            // Add active classes to AI Recommendations
            aiTab.classList.add('active');
            aiContent.classList.add('show', 'active');
            aiContent.classList.remove('fade');
        }
    }
});
