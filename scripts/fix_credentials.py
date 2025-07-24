#!/usr/bin/env python3
"""
Interactive credential fix script for cTrader API
This script will guide you through the OAuth process and update your .env file
"""

import requests
import json
import os
from dotenv import load_dotenv, set_key

def fix_credentials():
    """Interactive credential fix process"""
    
    print("=" * 80)
    print("CTRADER CREDENTIAL FIX - INTERACTIVE GUIDE")
    print("=" * 80)
    
    # Current credentials
    client_id = "12066_pp7glRUIsDmFaeXcBzMljuAz4083XEzL4GnegdcxiVKhshAXDt"
    client_secret = "lbcjnJTdbd1I4QCeSArlQyh4zN8r84EnU78idxHzrbpYhHvGlv"
    
    print("🔧 STEP 1: Generate New Authorization URL")
    print("=" * 50)
    auth_url = f"https://connect.icmarkets.com/oauth/authorize?client_id={client_id}&redirect_uri=http://localhost/&scope=trading"
    print(f"🔗 Open this URL in your browser:")
    print(f"   {auth_url}")
    print()
    
    print("📋 STEP 2: Complete OAuth Flow")
    print("=" * 50)
    print("1. Log in with your IC Markets cTrader credentials")
    print("2. When prompted, SELECT YOUR DEMO ACCOUNT")
    print("3. Grant 'trading' permissions")
    print("4. After authorization, you'll be redirected to http://localhost/")
    print("5. Copy the authorization code from the URL (after '?code=')")
    print()
    
    auth_code = input("📝 Enter the authorization code: ").strip()
    
    if not auth_code:
        print("❌ No authorization code provided")
        return
    
    print("🔄 STEP 3: Exchange Code for New Tokens")
    print("=" * 50)
    
    try:
        response = requests.post(
            "https://connect.icmarkets.com/api/v2/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": auth_code,
                "redirect_uri": "http://localhost/",
                "client_id": client_id,
                "client_secret": client_secret,
            }
        )
        
        if response.status_code == 200:
            token_data = response.json()
            new_access_token = token_data['access_token']
            new_refresh_token = token_data['refresh_token']
            
            print("✅ New tokens received successfully!")
            print(f"   Access Token: {new_access_token[:20]}...")
            print(f"   Refresh Token: {new_refresh_token[:20]}...")
            
            # Update .env file
            env_path = ".env"
            set_key(env_path, "CTRADER_ACCESS_TOKEN", new_access_token)
            set_key(env_path, "CTRADER_REFRESH_TOKEN", new_refresh_token)
            
            print("📝 STEP 4: .env file updated")
            print("=" * 50)
            print("✅ Your .env file has been updated with new tokens")
            print("   You can now run the verification tool")
            print()
            print("🚀 NEXT STEPS:")
            print("   1. Run: python scripts/test_real_credentials.py")
            print("   2. Then run: python scripts/verify_microstructure_real.py")
            
        else:
            print(f"❌ Token exchange failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error during token exchange: {e}")

if __name__ == "__main__":
    fix_credentials()
