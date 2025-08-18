require("@nomiclabs/hardhat-ethers");

module.exports = {
  solidity: "0.8.19",
  networks: {
    mainnet: {
      url: "https://eth-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
      accounts: ["0x" + process.env.PRIVATE_KEY]
    },
    polygon: {
      url: "https://polygon-mainnet.g.alchemy.com/v2/alcht_oZ7wU7JpIoZejlOWUcMFOpNsIlLDsX",
      accounts: ["0x" + process.env.PRIVATE_KEY]
    }
  }
};
