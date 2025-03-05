class DisableCSPMiddleware:
    """Middleware that completely disables Content Security Policy.
    
    WARNING: This should only be used in development!
    """
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Remove any existing CSP headers
        for header in list(response.headers.keys()):
            if 'content-security-policy' in header.lower():
                del response[header]
        
        # Add a permissive CSP header
        response['Content-Security-Policy'] = "default-src * 'unsafe-inline' 'unsafe-eval'; script-src * 'unsafe-inline' 'unsafe-eval'; connect-src * 'unsafe-inline'; img-src * data: blob: 'unsafe-inline'; frame-src *; style-src * 'unsafe-inline';"
        
        return response


class DevelopmentCSPMiddleware:
    """Middleware that sets a permissive Content Security Policy for development environments.
    
    This middleware is less extreme than DisableCSPMiddleware, allowing for a more
    targeted approach to CSP in development environments.
    
    WARNING: This should only be used in development!
    """
    
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        
        # Set a development-friendly CSP that still provides some security
        csp_directives = {
            'default-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'", "data:", "blob:"],
            'script-src': ["'self'", "'unsafe-inline'", "'unsafe-eval'", "cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com"],
            'style-src': ["'self'", "'unsafe-inline'", "fonts.googleapis.com", "cdn.jsdelivr.net", "cdnjs.cloudflare.com"],
            'img-src': ["'self'", "data:", "blob:"],
            'font-src': ["'self'", "fonts.gstatic.com", "cdn.jsdelivr.net", "cdnjs.cloudflare.com"],
            'connect-src': ["'self'", "ws:", "wss:", "blob:"],
            'worker-src': ["'self'", "blob:"],
            'frame-src': ["'self'"]
        }
        
        # Build the CSP header value
        csp_value = '; '.join([
            f"{directive} {' '.join(sources)}"
            for directive, sources in csp_directives.items()
        ])
        
        # Set the CSP header
        response['Content-Security-Policy'] = csp_value
        
        return response
