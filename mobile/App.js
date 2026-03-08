import React, { useState, useEffect, useRef } from 'react';
import { Text, View, TextInput, TouchableOpacity, SafeAreaView, Alert } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import * as Notifications from 'expo-notifications';
import { setBackendUrl, getBackendUrl, sendSensorData, registerPushToken } from './src/api';

const BATCH_SIZE = 20;

export default function App() {
  const [deviceId, setDeviceId] = useState('mobile-user-1');
  const [serverUrl, setServerUrl] = useState(getBackendUrl());
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [fallDetected, setFallDetected] = useState(false);
  const [accData, setAccData] = useState({ x: 0, y: 0, z: 0 });
  const [gyroData, setGyroData] = useState({ x: 0, y: 0, z: 0 });
  const [status, setStatus] = useState('Idle');
  const [pushToken, setPushToken] = useState('');

  const subscriptionRef = useRef(null);
  const gyroSubscriptionRef = useRef(null);
  const latestGyroRef = useRef({ x: 0, y: 0, z: 0 });
  const dataBatchRef = useRef({ acc: [], gyro: [] });

  useEffect(() => {
    registerForPushNotifications();
    // Clean up subscription on unmount
    return () => stopMonitoring();
  }, []);

  const startMonitoring = async () => {
    if (!serverUrl || !deviceId) {
      Alert.alert('Error', 'Please enter Device ID and Backend URL');
      return;
    }

    setBackendUrl(serverUrl);
    setFallDetected(false);
    setStatus('Monitoring... Starting sensors');
    setIsMonitoring(true);
    dataBatchRef.current = { acc: [], gyro: [] };

    if (pushToken) {
      try {
        await registerPushToken(deviceId, pushToken);
      } catch (error) {
        console.warn('Push registration failed:', error);
      }
    }

    // Set update interval (e.g., 50ms = 20Hz)
    Accelerometer.setUpdateInterval(50);
    Gyroscope.setUpdateInterval(50);

    subscriptionRef.current = Accelerometer.addListener(processAccelerometerData);
    gyroSubscriptionRef.current = Gyroscope.addListener(processGyroscopeData);
    setStatus('Monitoring Active');
  };

  const stopMonitoring = () => {
    if (subscriptionRef.current) {
      subscriptionRef.current.remove();
      subscriptionRef.current = null;
    }
    if (gyroSubscriptionRef.current) {
      gyroSubscriptionRef.current.remove();
      gyroSubscriptionRef.current = null;
    }
    setIsMonitoring(false);
    setStatus('Idle');
    dataBatchRef.current = { acc: [], gyro: [] };
  };

  const processAccelerometerData = (data) => {
    setAccData(data);
    const gyro = latestGyroRef.current;
    dataBatchRef.current.acc.push([data.x, data.y, data.z]);
    dataBatchRef.current.gyro.push([gyro.x, gyro.y, gyro.z]);

    if (dataBatchRef.current.acc.length >= BATCH_SIZE) {
      const batchToSend = { ...dataBatchRef.current };
      dataBatchRef.current = { acc: [], gyro: [] }; // Reset batch
      sendDataToBackend(batchToSend.acc, batchToSend.gyro);
    }
  };

  const processGyroscopeData = (data) => {
    setGyroData(data);
    latestGyroRef.current = data;
  };

  const sendDataToBackend = async (accBatch, gyroBatch) => {
    try {
      // Small UI update to show activity without disrupting the main state
      const response = await sendSensorData(deviceId, accBatch, gyroBatch);
      
      if (response && response.fall) {
        handleFallDetection();
      }
    } catch (error) {
      setStatus('Error connecting to backend');
      console.warn("Backend Error:", error);
    }
  };

  const registerForPushNotifications = async () => {
    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;

    if (existingStatus !== 'granted') {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }

    if (finalStatus !== 'granted') {
      return;
    }

    const tokenResponse = await Notifications.getExpoPushTokenAsync();
    setPushToken(tokenResponse.data);
  };

  const handleFallDetection = () => {
    setFallDetected(true);
    setStatus('Fall Detected!');
    stopMonitoring();
  };

  const toggleMonitoring = () => {
    if (isMonitoring) {
      stopMonitoring();
    } else {
      startMonitoring();
    }
  };

  return (
    <SafeAreaView className="flex-1 bg-neutral-100 justify-center items-center">
      {fallDetected ? (
        <View className="w-11/12 bg-red-500 rounded-2xl p-6 shadow-md items-center">
          <Text className="text-3xl font-bold text-white mb-3 text-center">⚠️ FALL DETECTED ⚠️</Text>
          <Text className="text-lg text-white mb-8">Assistance may be needed.</Text>
          <TouchableOpacity className="bg-white px-8 py-4 rounded-full" onPress={() => setFallDetected(false)}>
            <Text className="text-red-500 text-base font-bold">Reset Status</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <View className="w-11/12 bg-white rounded-2xl p-6 shadow-md">
          <Text className="text-2xl font-bold mb-5 text-center text-neutral-800">Fall Detection Monitor</Text>
          
          <View className="mb-4">
            <Text className="text-sm font-semibold text-neutral-600 mb-1.5">Device ID</Text>
            <TextInput
              className="border border-neutral-300 rounded-lg p-3 text-base bg-neutral-50"
              value={deviceId}
              onChangeText={setDeviceId}
              placeholder="Enter Device ID"
              editable={!isMonitoring}
            />
          </View>

          <View className="mb-4">
            <Text className="text-sm font-semibold text-neutral-600 mb-1.5">Backend URL</Text>
            <TextInput
              className="border border-neutral-300 rounded-lg p-3 text-base bg-neutral-50"
              value={serverUrl}
              onChangeText={setServerUrl}
              placeholder="http://192.168.1.x:8080"
              autoCapitalize="none"
              keyboardType="url"
              editable={!isMonitoring}
            />
          </View>

          <View className="bg-neutral-50 p-4 rounded-lg mb-5 border border-neutral-200">
            <Text className="text-sm font-semibold text-neutral-600 mb-1.5">Accelerometer Live Data:</Text>
            <Text className="font-mono text-base text-neutral-800 my-0.5">X: {accData.x.toFixed(3)}</Text>
            <Text className="font-mono text-base text-neutral-800 my-0.5">Y: {accData.y.toFixed(3)}</Text>
            <Text className="font-mono text-base text-neutral-800 my-0.5">Z: {accData.z.toFixed(3)}</Text>
          </View>

          <View className="bg-neutral-50 p-4 rounded-lg mb-5 border border-neutral-200">
            <Text className="text-sm font-semibold text-neutral-600 mb-1.5">Gyroscope Live Data:</Text>
            <Text className="font-mono text-base text-neutral-800 my-0.5">X: {gyroData.x.toFixed(3)}</Text>
            <Text className="font-mono text-base text-neutral-800 my-0.5">Y: {gyroData.y.toFixed(3)}</Text>
            <Text className="font-mono text-base text-neutral-800 my-0.5">Z: {gyroData.z.toFixed(3)}</Text>
          </View>

          <Text className="text-sm text-neutral-600 text-center mb-5 italic">Status: {status}</Text>

          <TouchableOpacity 
            className={`p-4 rounded-lg items-center ${isMonitoring ? 'bg-red-500' : 'bg-green-500'}`} 
            onPress={toggleMonitoring}
          >
            <Text className="text-white text-base font-bold">
              {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
            </Text>
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
  );
}
