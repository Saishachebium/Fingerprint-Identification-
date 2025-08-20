# Fingerprint Identification

![MIT License](https://img.shields.io/badge/license-MIT-green)  

This project identifies whether a user’s fingerprint matches an existing record. It uses algorithms such as the Zhang–Suen thinning algorithm and recursive logic to analyze ridge patterns and compare them against stored data.

## Features

- Accurate fingerprint matching using Zhang–Suen thinning and recursive analysis.  
- Handles real-world factors like lighting, finger positioning, and ridge clarity.  
- Simple sign-up and login workflow for registering and verifying fingerprints.  
- Lightweight and easy to run using standard Python tools.

## How It Works

The program takes a fingerprint image from the user and processes it to detect ridge patterns. It then compares these patterns to the stored fingerprint records to determine a match.

## Factors Affecting Accuracy

The reliability of fingerprint identification depends on several factors:

- **Webcam or scanner quality** – Higher resolution images produce clearer ridge detection.  
- **Lighting conditions** – Shadows or glare can distort the fingerprint image.  
- **Finger positioning** – Rotation or shifting of the finger can affect the match.  
- **Ridge clarity** – Smudges, dryness, or excessive pressure can reduce accuracy.

## Usage

### Sign Up

1. Run the program and press **2** to sign up.  
2. Enter your name (don’t press Enter yet).  
3. Position your finger in the red box and make sure your ridges are clearly visible (e.g., slightly colored or high-contrast for the camera).  
4. Press **Enter** to register your fingerprint.

### Log In

1. Run the program and press **1** to log in.  
2. Place your finger in the red box.  
3. Wait for the program to indicate whether your fingerprint matches the registered one.

> **Note:** Fingerprint matching is not 100% accurate. Factors such as webcam quality, lighting, finger positioning, and ridge clarity can affect results.

## Future Improvements

- Improve matching accuracy by incorporating multiple images per user.  
- Add support for multiple simultaneous users.  
- Develop a GUI for easier use.  
- Enhance preprocessing for better handling of low-quality images.

## License

This project is licensed under the MIT License.  
