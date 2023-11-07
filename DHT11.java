package FinalProject;

import com.pi4j.io.gpio.GpioController;
import com.pi4j.io.gpio.GpioFactory;
import com.pi4j.io.gpio.GpioPinDigitalOutput;
import com.pi4j.io.gpio.GpioPinPwmOutput;
import com.pi4j.io.gpio.PinState;
import com.pi4j.io.gpio.RaspiPin;
import com.pi4j.wiringpi.Gpio;
import com.pi4j.wiringpi.GpioUtil;

import week9.MCP3204;

public class DHT11 {
	private static final int MAXTIMINGS = 85; // Data 교환이 이루어질 수 있는 최대 시간 정의
    private final int[] dht11_f = {0, 0, 0, 0, 0}; //dht11 data format (5 bytes)    
    GpioController gpio = GpioFactory.getInstance();
	GpioPinPwmOutput pwm = gpio.provisionSoftPwmOutputPin(RaspiPin.GPIO_16);
    private boolean po = true;//전원 온오프 상태를 나타내주는 변수 초기값은 켜져 있는 것으로 한다.
	
	
    public DHT11() {
    	// Setup wiringPi
    	if (Gpio.wiringPiSetup() == -1) {
    		System.out.println(" ==>> GPIO SETUP FAILED");
    		return;
    	}
    }

    public float[] getData(final int pin) {//현재 온도와 습도 정보를 가져오는 메소드다
    	int laststate = Gpio.HIGH; // Signal 상태 변화를 알기 위해 기존 상태 저장
    	int j = 0; // 수신한 Bit의 index counter
    	float h = -99; // 습도 (%)
    	float c = -99; // 섭씨 온도 (°C)
    	float f = -99; // 화씨 온도 (°F)

    	// Integral RH, Decimal RH, Integral T, Decimal T, Checksum
    	dht11_f[0] = dht11_f[1] = dht11_f[2] = dht11_f[3] = dht11_f[4] = 0;    	

    	// 1. DHT11 센서에게 start signal 전달
    	Gpio.pinMode(pin, Gpio.OUTPUT);
    	Gpio.digitalWrite(pin, Gpio.LOW);
    	Gpio.delay(18); // 18 ms

    	// 2. Pull-up -> 수신 모드로 전환 -> 센서의 응답 대기
    	Gpio.digitalWrite(pin, Gpio.HIGH);
    	Gpio.pinMode(pin, Gpio.INPUT);

    	// 3. 센서의 응답에 따른 동작
    	for (int i = 0; i < MAXTIMINGS; i++) {
    		int counter = 0;
    		while (Gpio.digitalRead(pin) == laststate) { // GPIO pin 상태가 바뀌지 않으면 대기
    			counter++;
    			Gpio.delayMicroseconds(1);
    			if (counter == 255) {
    				break;
    			}
    		}
    		laststate = Gpio.digitalRead(pin);
    		if (counter == 255) {
    			break;
    		}

    		// 각각의 bit 데이터 저장
    		if (i >= 4 && i % 2 == 0) { // 첫 4개의 상태 변화는 무시, laststate 가 low에서 high로 바뀔때만 값 저장
    			// Data 저장
    			dht11_f[j / 8] <<= 1; // 0 bit 
    			if (counter > 16) {
    				dht11_f[j / 8] |= 1; // 1 bit
    			}
    			j++;
    		}
    	}

    	// Checksum 확인
    	// Check we read 40 bits (8bit x 5 ) + verify checksum in the last
    	if (j >= 40 && getChecksum()) {
    		h = (float) ((dht11_f[0] << 8) + dht11_f[1]) / 10;
    		if (h > 100) {
    			h = dht11_f[0]; // for DHT11
    		}
    		c = (float) (((dht11_f[2] & 0x7F) << 8) + dht11_f[3]) / 10;
    		if (c > 125) {
    			c = dht11_f[2]; // for DHT11
    		}
    		if ((dht11_f[2] & 0x80) != 0) {
    			c = -c;
    		}
    		f = c * 1.8f + 32;
    		System.out.println("Humidity = " + h + "% Temperature = " + c + "°C | " + f + "°F)");
    	}
    	else {
    		System.out.println("Checksum Error");
    	}

    	float[] result = {h,c,f};
    	
    	this.motor(c);//모터제어 메소드 호출
    	
    	return result;        
    }

    private boolean getChecksum() {
    	return dht11_f[4] == (dht11_f[0] + dht11_f[1] + dht11_f[2] + dht11_f[3] & 0xFF);
    }
    public boolean power_re() {
    	return this.po;

    }
    public void motor(float c) {//현재 온도를 고려해 전원을 
    	
    	pwm.setPwmRange(200);
    	
    		if(c >= 27 && po) {//실내적정온도를 27도로 가정한다. 실제는 다르다. 테스트를 위해 보편적인 온도로 설정했다.
    			try {
        			pwm.setPwm(24);
        			Thread.sleep(1000);
        			pwm.setPwm(6);
        			Thread.sleep(1000);
        			
        			
        		}
        		catch (Exception e) {
        			
        		}
    			this.po = !this.po;
    		}else if(c < 27 && !po) {//적정온도보다 현재 온도가 낮을 떄 동시에 전원이 꺼져있을 때 전원을 키게 된다.
    			try {
        			pwm.setPwm(24);
        			Thread.sleep(1000);
        			pwm.setPwm(6);
        			Thread.sleep(1000);
        			
        			
        		}
        		catch (Exception e) {
        			
        		}
    			this.po = !this.po;
    		}
    	
    }
    
    
}
