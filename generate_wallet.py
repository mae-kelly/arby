from eth_account import Account
import secrets

# Generate a new Ethereum wallet
private_key = "0x" + secrets.token_hex(32)
account = Account.from_key(private_key)

print("🔑 NEW ETHEREUM WALLET GENERATED")
print("=" * 50)
print(f"Private Key: {private_key}")
print(f"Address: {account.address}")
print("=" * 50)
print("⚠️  SAVE THIS PRIVATE KEY SECURELY!")
print("💰 Send some ETH to this address for gas fees")