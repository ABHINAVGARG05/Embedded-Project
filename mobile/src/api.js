import axios from "axios";

// The default backend port is 8080 based on the analysis
// This will be configurable in the UI
let backendUrl = "http://192.168.1.100:8080";

export const setBackendUrl = (url) => {
  backendUrl = url;
};

export const getBackendUrl = () => backendUrl;

export const sendSensorData = async (deviceId, accData, gyroData) => {
  try {
    const response = await axios.post(`${backendUrl}/api/sensor`, {
      device_id: deviceId,
      acc: accData,
      gyro: gyroData,
    });
    return response.data;
  } catch (error) {
    console.error("API Error:", error.message);
    throw error;
  }
};

export const registerPushToken = async (deviceId, expoToken) => {
  try {
    const response = await axios.post(`${backendUrl}/api/push/register`, {
      device_id: deviceId,
      expo_token: expoToken,
    });
    return response.data;
  } catch (error) {
    console.error("Push registration error:", error.message);
    throw error;
  }
};
