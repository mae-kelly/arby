require("@nomiclabs/hardhat-ethers");
require("dotenv").config();

module.exports = {
  solidity: "0.8.19",
  networks: {
    goerli: {
      url: process.env.GOERLI_RPC,
      accounts: [process.env.TESTNET_PRIVATE_KEY],
      chainId: 5,
      gasPrice: "auto"
    },
    sepolia: {
      url: process.env.SEPOLIA_RPC,
      accounts: [process.env.TESTNET_PRIVATE_KEY],
      chainId: 11155111,
      gasPrice: "auto"
    },
    mumbai: {
      url: process.env.MUMBAI_RPC,
      accounts: [process.env.TESTNET_PRIVATE_KEY],
      chainId: 80001,
      gasPrice: "auto"
    }
  },
  paths: {
    sources: "./contracts",
    tests: "./test",
    cache: "./cache",
    artifacts: "./artifacts"
  }
};
