#!/usr/bin/env python3
"""
Simple proxy server for PyPI Stats API to handle CORS issues
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.request
import urllib.parse
import time
from urllib.error import HTTPError, URLError

class PyPIProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the URL to get the library name
        if self.path.startswith('/api/packages/'):
            parts = self.path.split('/')
            if len(parts) >= 4:
                library = parts[3]
                endpoint = parts[4] if len(parts) > 4 else 'recent'
                
                try:
                    # Make request to PyPI Stats API
                    url = f"https://pypistats.org/api/packages/{library}/{endpoint}"
                    
                    # Add headers to avoid being blocked
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (compatible; PyPIStatsProxy/1.0)',
                        'Accept': 'application/json'
                    }
                    
                    request = urllib.request.Request(url, headers=headers)
                    
                    # Add delay to avoid rate limiting
                    time.sleep(0.5)
                    
                    with urllib.request.urlopen(request, timeout=10) as response:
                        data = response.read()
                        
                        # Send successful response
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
                        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                        self.end_headers()
                        self.wfile.write(data)
                        
                except HTTPError as e:
                    # Handle HTTP errors
                    error_data = {
                        "error": f"HTTP Error {e.code}",
                        "message": str(e.reason),
                        "library": library
                    }
                    self.send_error_response(e.code, error_data)
                    
                except URLError as e:
                    # Handle URL errors
                    error_data = {
                        "error": "Network Error",
                        "message": str(e.reason),
                        "library": library
                    }
                    self.send_error_response(500, error_data)
                    
                except Exception as e:
                    # Handle other errors
                    error_data = {
                        "error": "Server Error",
                        "message": str(e),
                        "library": library
                    }
                    self.send_error_response(500, error_data)
            else:
                self.send_error_response(400, {"error": "Invalid API path"})
        else:
            self.send_error_response(404, {"error": "Not found"})
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_error_response(self, code, error_data):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(error_data).encode())
    
    def log_message(self, format, *args):
        # Custom logging
        print(f"[{self.log_date_time_string()}] {format % args}")

def run_server(port=8001):
    server_address = ('', port)
    httpd = HTTPServer(server_address, PyPIProxyHandler)
    print(f"PyPI proxy server running on http://localhost:{port}")
    print("API endpoint: http://localhost:8001/api/packages/[library]/recent")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    run_server()