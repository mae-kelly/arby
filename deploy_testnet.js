const { ethers } = require("hardhat");

async function main() {
    console.log("🧪 Deploying to TESTNET...");
    
    // Deploy the simulation contract
    const TestnetFlashLoan = await ethers.getContractFactory("TestnetFlashLoanSimulator");
    const contract = await TestnetFlashLoan.deploy();
    
    await contract.deployed();
    
    console.log("✅ Testnet contract deployed to:", contract.address);
    console.log("🚰 Use giveMeMoney() function to get test tokens");
    
    // Give deployer some test tokens
    const tx = await contract.giveMeMoney();
    await tx.wait();
    
    console.log("✅ Test tokens minted to deployer");
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
