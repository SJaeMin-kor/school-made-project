package FinalProject;


import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.net.InetAddress;
import java.net.UnknownHostException;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

import org.ws4d.coap.core.CoapClient;
import org.ws4d.coap.core.CoapConstants;
import org.ws4d.coap.core.connection.BasicCoapChannelManager;
import org.ws4d.coap.core.connection.api.CoapChannelManager;
import org.ws4d.coap.core.connection.api.CoapClientChannel;
import org.ws4d.coap.core.enumerations.CoapMediaType;
import org.ws4d.coap.core.enumerations.CoapRequestCode;
import org.ws4d.coap.core.messages.api.CoapRequest;
import org.ws4d.coap.core.messages.api.CoapResponse;
import org.ws4d.coap.core.rest.CoapData;
import org.ws4d.coap.core.tools.Encoder;

public class GUI_Client extends JFrame implements CoapClient{
	private static final boolean exitAfterResponse = false;
	JButton btn_get = new JButton("GET");
	JButton btn_post = new JButton("POST");
	JButton btn_put = new JButton("PUT");
	JButton btn_delete = new JButton("DELETE");
	
	JButton btn_observe = new JButton("OBSERVE");
	
	JLabel path_label = new JLabel("Path");
	JTextArea path_text = new JTextArea("/.well-known/core", 1,1);//스크롤바 없음
	JLabel payload_label = new JLabel("Payload");
	JTextArea payload_text = new JTextArea("", 1,1);//스크롤바 없음
	static JTextArea display_text = new JTextArea();
	JScrollPane display_text_jp  = new JScrollPane(display_text);
	JLabel display_label = new JLabel("Display");
	
	CoapClientChannel clientChannel = null;
	
	//DHT11 dh = new DHT11();
	
	
	public GUI_Client (String serverAddress, int serverPort) {
		//제목 설정
		super("임베디드 실습 GUI client");
		//레이아웃 설정
		this.setLayout(null);
		String sAddress = serverAddress;
		int sPort = serverPort;

		CoapChannelManager channelManager = BasicCoapChannelManager.getInstance();

		

		try {
			clientChannel = channelManager.connect(this, InetAddress.getByName(sAddress), sPort);
		} catch (UnknownHostException e) {
			e.printStackTrace();
			System.exit(-1);
		}


		if (null == clientChannel) {
			return;
		}

		//btn
		btn_get.setBounds(20, 670, 100, 50);
		btn_put.setBounds(130, 670, 100, 50);
		btn_post.setBounds(240, 670, 100, 50);
		btn_delete.setBounds(350, 670, 100, 50);
		btn_observe.setBounds(460, 670, 100, 50);
		
		btn_get.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				String path = path_text.getText();
				String payload = payload_text.getText();
				CoapRequest request = clientChannel.createRequest(CoapRequestCode.GET, path, true);
				displayRequest(request);
				clientChannel.sendMessage(request);
			}
		});
		btn_put.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				String path = path_text.getText();
				String payload = payload_text.getText();
				CoapRequest request = clientChannel.createRequest(CoapRequestCode.PUT, path, true);
				request.setPayload(new CoapData(payload, CoapMediaType.text_plain));
				displayRequest(request);
				clientChannel.sendMessage(request);
			}
		});
		btn_post.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				String path = path_text.getText();
				String payload = payload_text.getText();
				CoapRequest request = clientChannel.createRequest(CoapRequestCode.POST, path, true);
				request.setPayload(new CoapData(payload, CoapMediaType.text_plain));
				displayRequest(request);
				clientChannel.sendMessage(request);
			}
		});
		btn_delete.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				String path = path_text.getText();
				String payload = payload_text.getText();
				CoapRequest request = clientChannel.createRequest(CoapRequestCode.DELETE, path, true);
				displayRequest(request);
				clientChannel.sendMessage(request);
			}
		});
		btn_observe.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				// TODO Auto-generated method stub
				String path = path_text.getText();
				String payload = payload_text.getText();
				CoapRequest request = clientChannel.createRequest(CoapRequestCode.GET, path, true);
				request.setToken(Encoder.StringToByte("obToken"));
				request.setObserveOption(0);
				displayRequest(request);
				clientChannel.sendMessage(request);
			}
		});
		
		
		
		payload_label.setBounds(20, 570, 350, 30);
		payload_text.setBounds(20, 600, 440, 30);
		payload_text.setFont(new Font("arian", Font.BOLD, 15));
		
		path_label.setBounds(20, 500, 350, 30);
		path_text.setBounds(20, 530, 440, 30);
		path_text.setFont(new Font("arian", Font.BOLD, 15));
		
		display_label.setBounds(20, 10, 100, 20);
		display_text.setLineWrap(true);
		display_text.setFont(new Font("arian", Font.BOLD, 15));
		display_text_jp.setBounds(20, 40, 740, 430);
		
				
		this.add(btn_get);
		this.add(btn_post);
		this.add(btn_put);
		this.add(btn_delete);
		this.add(btn_observe);
		this.add(path_text);
		this.add(path_label);
		this.add(payload_label);
		this.add(payload_text);
		this.add(display_text_jp);
		this.add(display_label);

		//프레임 크기 지정	
		this.setSize(800, 800);

		//프레임 보이기
		this.setVisible(true);

		//swing에만 있는 X버튼 클릭시 종료
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

	@Override
	public void onConnectionFailed(CoapClientChannel channel, boolean notReachable, boolean resetByServer) {
		System.out.println("Connection Failed");
		System.exit(-1);
	}

	@Override
	public void onResponse(CoapClientChannel channel, CoapResponse response) {
		if (response.getPayload() != null) {
			display_text.append(
					"Response: " + response.toString() + " payload: " + Encoder.ByteToString(response.getPayload()));
			display_text.setCaretPosition(display_text.getDocument().getLength());  
		} else {
			display_text.append("Response: " + response.toString());
			display_text.setCaretPosition(display_text.getDocument().getLength());
		}
		if (GUI_Client.exitAfterResponse) {
			display_text.append("===END===");
			System.exit(0);
		}
		display_text.append(System.lineSeparator());
		display_text.append("*");
		display_text.append(System.lineSeparator());
		
		if(Encoder.ByteToString(response.getToken()).equals("obToken")) {
			float temp = Float.parseFloat(Encoder.ByteToString(response.getPayload()));
			this.control_led(temp);
		}
	}
	
	public void control_led(float temp) {
		CoapRequest request = clientChannel.createRequest(CoapRequestCode.PUT, "/led", true);
		if(temp > 24.0) {
			request.setPayload(new CoapData("red", CoapMediaType.text_plain));
		}else {
			request.setPayload(new CoapData("green", CoapMediaType.text_plain));
		}
	}

	@Override
	public void onMCResponse(CoapClientChannel channel, CoapResponse response, InetAddress srcAddress, int srcPort) {
		// TODO Auto-generated method stub
	}
		
	private void displayRequest(CoapRequest request){
		if(request.getPayload() != null){
			display_text.append("Request: "+request.toString() + " payload: " + Encoder.ByteToString(request.getPayload()) + " resource: " + request.getUriPath());
			display_text.setCaretPosition(display_text.getDocument().getLength());  
		} 
		else{
			display_text.append("Request: "+request.toString() + " resource: " + request.getUriPath());
			display_text.setCaretPosition(display_text.getDocument().getLength());  
		}
		display_text.append(System.lineSeparator());
		display_text.append("*");
		
		
		display_text.append(System.lineSeparator());
	}
	

	public static void main(String[] args){
		//프레임 열기
		GUI_Client gui = new GUI_Client("fe80::3903:9e4a:55bd:d568", CoapConstants.COAP_DEFAULT_PORT);
		
		
		
	}
	
	
}