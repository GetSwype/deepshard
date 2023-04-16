pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/token/ERC20/ERC20.sol";

contract AstraToken is ERC20 {
    address public minter;

    constructor(address _minter) ERC20("Astra", "AST") {
        // Set the address of the authorized minter
        minter = _minter;
    }

    // Minting function that can only be called by the authorized minter
    function mint(address to, uint256 amount) external {
        require(msg.sender == minter, "Only the authorized minter can mint tokens");
        _mint(to, amount);
    }

    // Update the minter to a new address (optional)
    function updateMinter(address newMinter) external {
        require(msg.sender == minter, "Only the current minter can update the minter");
        minter = newMinter;
    }
}
