/**
 * AskYourDoc - Universal Embeddable Chat Bubble Widget
 * Allows anyone to embed AskYourDoc into their website with a single <script> tag.
 *
 * Usage:
 * <script src="https://your-domain.com/static/widget.js" 
 *         data-server="https://your-domain.com"
 *         data-title="Ask Document"
 *         data-position="bottom-right">
 * </script>
 */
(function() {
    'use strict';

    // Prevent duplicate initialization
    if (window.__askYourDocWidgetInitialized) return;
    window.__askYourDocWidgetInitialized = true;

    // Detect config from script tag
    const currentScript = document.currentScript || document.querySelector('script[src*="widget.js"]');
    const serverUrl = (currentScript && currentScript.getAttribute('data-server')) 
        ? currentScript.getAttribute('data-server').replace(/\/+$/, '')
        : (currentScript ? new URL(currentScript.src).origin : window.location.origin);

    const position = (currentScript && currentScript.getAttribute('data-position')) || 'bottom-right';
    const widgetTitle = (currentScript && currentScript.getAttribute('data-title')) || 'Ask Document';

    // Inject styles
    const style = document.createElement('style');
    style.innerHTML = `
        .ayd-widget-launcher {
            position: fixed;
            bottom: 24px;
            ${position === 'bottom-left' ? 'left: 24px;' : 'right: 24px;'}
            width: 58px;
            height: 58px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.45), 0 2px 8px rgba(0, 0, 0, 0.3);
            border: none;
            cursor: pointer;
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s ease;
            outline: none;
        }
        .ayd-widget-launcher:hover {
            transform: scale(1.08) translateY(-2px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
        }
        .ayd-widget-launcher:active {
            transform: scale(0.96);
        }
        .ayd-widget-launcher svg {
            transition: transform 0.2s ease, opacity 0.2s ease;
        }
        .ayd-widget-container {
            position: fixed;
            bottom: 96px;
            ${position === 'bottom-left' ? 'left: 24px;' : 'right: 24px;'}
            width: 420px;
            height: 640px;
            max-width: calc(100vw - 32px);
            max-height: calc(100vh - 120px);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.1);
            background: #0a0d14;
            z-index: 999998;
            opacity: 0;
            transform: scale(0.85) translateY(20px);
            pointer-events: none;
            transform-origin: ${position === 'bottom-left' ? 'bottom left' : 'bottom right'};
            transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.2s ease;
        }
        .ayd-widget-container.ayd-open {
            opacity: 1;
            transform: scale(1) translateY(0);
            pointer-events: auto;
        }
        .ayd-widget-iframe {
            width: 100%;
            height: 100%;
            border: none;
            display: block;
        }
        @media (max-width: 480px) {
            .ayd-widget-container {
                bottom: 0 !important;
                right: 0 !important;
                left: 0 !important;
                top: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                max-width: 100vw !important;
                max-height: 100vh !important;
                border-radius: 0 !important;
            }
        }
    `;
    document.head.appendChild(style);

    // Create Launcher Button
    const launcher = document.createElement('button');
    launcher.className = 'ayd-widget-launcher';
    launcher.setAttribute('aria-label', widgetTitle);
    launcher.setAttribute('title', widgetTitle);

    const chatIcon = `
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
    `;
    const closeIcon = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
    `;
    launcher.innerHTML = chatIcon;

    // Create Iframe Container
    const container = document.createElement('div');
    container.className = 'ayd-widget-container';

    let iframeLoaded = false;
    let isOpen = false;

    function toggleWidget() {
        isOpen = !isOpen;
        if (isOpen) {
            if (!iframeLoaded) {
                const iframe = document.createElement('iframe');
                iframe.className = 'ayd-widget-iframe';
                iframe.src = `${serverUrl}/embed`;
                iframe.setAttribute('allow', 'clipboard-write');
                container.appendChild(iframe);
                iframeLoaded = true;
            }
            container.classList.add('ayd-open');
            launcher.innerHTML = closeIcon;
        } else {
            container.classList.remove('ayd-open');
            launcher.innerHTML = chatIcon;
        }
    }

    launcher.addEventListener('click', toggleWidget);

    // Mount to DOM
    document.body.appendChild(container);
    document.body.appendChild(launcher);
})();

