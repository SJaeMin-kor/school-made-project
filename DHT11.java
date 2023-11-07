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
	private static final int MAXTIMINGS = 85; // Data ��ȯ�� �̷���� �� �ִ� �ִ� �ð� ����
    private final int[] dht11_f = {0, 0, 0, 0, 0}; //dht11 data format (5 bytes)    
    GpioController gpio = GpioFactory.getInstance();
	GpioPinPwmOutput pwm = gpio.provisionSoftPwmOutputPin(RaspiPin.GPIO_16);
    private boolean po = true;//���� �¿��� ���¸� ��Ÿ���ִ� ���� �ʱⰪ�� ���� �ִ� ������ �Ѵ�.
	
	
    public DHT11() {
    	// Setup wiringPi
    	if (Gpio.wiringPiSetup() == -1) {
    		System.out.println(" ==>> GPIO SETUP FAILED");
    		return;
    	}
    }

    public float[] getData(final int pin) {//���� �µ��� ���� ������ �������� �޼ҵ��
    	int laststate = Gpio.HIGH; // Signal ���� ��ȭ�� �˱� ���� ���� ���� ����
    	int j = 0; // ������ Bit�� index counter
    	float h = -99; // ���� (%)
    	float c = -99; // ���� �µ� (��C)
    	float f = -99; // ȭ�� �µ� (��F)

    	// Integral RH, Decimal RH, Integral T, Decimal T, Checksum
    	dht11_f[0] = dht11_f[1] = dht11_f[2] = dht11_f[3] = dht11_f[4] = 0;    	

    	// 1. DHT11 �������� start signal ����
    	Gpio.pinMode(pin, Gpio.OUTPUT);
    	Gpio.digitalWrite(pin, Gpio.LOW);
    	Gpio.delay(18); // 18 ms

    	// 2. Pull-up -> ���� ���� ��ȯ -> ������ ���� ���
    	Gpio.digitalWrite(pin, Gpio.HIGH);
    	Gpio.pinMode(pin, Gpio.INPUT);

    	// 3. ������ ���信 ���� ����
    	for (int i = 0; i < MAXTIMINGS; i++) {
    		int counter = 0;
    		while (Gpio.digitalRead(pin) == laststate) { // GPIO pin ���°� �ٲ��� ������ ���
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

    		// ������ bit ������ ����
    		if (i >= 4 && i % 2 == 0) { // ù 4���� ���� ��ȭ�� ����, laststate �� low���� high�� �ٲ𶧸� �� ����
    			// Data ����
    			dht11_f[j / 8] <<= 1; // 0 bit 
    			if (counter > 16) {
    				dht11_f[j / 8] |= 1; // 1 bit
    			}
    			j++;
    		}
    	}

    	// Checksum Ȯ��
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
    		System.out.println("Humidity = " + h + "% Temperature = " + c + "��C | " + f + "��F)");
    	}
    	else {
    		System.out.println("Checksum Error");
    	}

    	float[] result = {h,c,f};
    	
    	this.motor(c);//�������� �޼ҵ� ȣ��
    	
    	return result;        
    }

    private boolean getChecksum() {
    	return dht11_f[4] == (dht11_f[0] + dht11_f[1] + dht11_f[2] + dht11_f[3] & 0xFF);
    }
    public boolean power_re() {
    	return this.po;

    }
    public void motor(float c) {//���� �µ��� ����� ������ 
    	
    	pwm.setPwmRange(200);
    	
    		if(c >= 27 && po) {//�ǳ������µ��� 27���� �����Ѵ�. ������ �ٸ���. �׽�Ʈ�� ���� �������� �µ��� �����ߴ�.
    			try {
        			pwm.setPwm(24);
        			Thread.sleep(1000);
        			pwm.setPwm(6);
        			Thread.sleep(1000);
        			
        			
        		}
        		catch (Exception e) {
        			
        		}
    			this.po = !this.po;
    		}else if(c < 27 && !po) {//�����µ����� ���� �µ��� ���� �� ���ÿ� ������ �������� �� ������ Ű�� �ȴ�.
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
