import java.awt.*;
import java.awt.event.*;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.Serial;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileNameExtensionFilter;

import uk.co.caprica.vlcj.player.component.EmbeddedMediaPlayerComponent;


public class App extends JFrame implements AWTEventListener, Runnable {
    @Serial
    private static final long serialVersionUID = 1L;
    private static final String TITLE = "Media Player";
    private final EmbeddedMediaPlayerComponent mediaPlayerComponent;
    private final JFileChooser fileChooser;
    private final FileNameExtensionFilter extensionFilter;
    private JButton playBtn, pauseBtn, selectFile, rewind, ffw;
    private JLabel pathName;
    private JPanel content, controls;


    public App(String title) {
        super(title);
        mediaPlayerComponent = new EmbeddedMediaPlayerComponent();
        fileChooser = new JFileChooser();
        extensionFilter = new FileNameExtensionFilter("MP4 & AVI videos", "mp4", "avi");
        fileChooser.setFileFilter(extensionFilter);
        this.getToolkit().addAWTEventListener(this, AWTEvent.KEY_EVENT_MASK);
    }

    public void init() {
        this.setBounds(100, 100, 700, 600);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                mediaPlayerComponent.release();
                System.exit(0);
            }
        });

        content = new JPanel();
        content.setLayout(new BorderLayout());
        content.add(mediaPlayerComponent, BorderLayout.CENTER);

        controls = new JPanel();
        pathName = new JLabel("Ready");
        playBtn = new JButton("Play");
        pauseBtn = new JButton("Pause");
        selectFile = new JButton("Open File");
        rewind = new JButton("Rewind");
        ffw = new JButton("Fast Forward");
        controls.add(pathName);
        controls.add(rewind);
        controls.add(playBtn);
        controls.add(pauseBtn);
        controls.add(selectFile);
        controls.add(ffw);
        content.add(controls, BorderLayout.SOUTH);

        playBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                mediaPlayerComponent.mediaPlayer().controls().play();
            }
        });

        pauseBtn.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                mediaPlayerComponent.mediaPlayer().controls().pause();
            }
        });

        selectFile.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (e.getSource() == selectFile) {
                    int rVal = fileChooser.showOpenDialog(App.this);

                    if (rVal == JFileChooser.APPROVE_OPTION) {
                        File file = fileChooser.getSelectedFile();
                        videoLoad(file.getAbsolutePath());
                        pathName.setText("Playing: " + file.getAbsolutePath());
                    }
                }
            }
        });

        rewind.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                mediaPlayerComponent.mediaPlayer().controls().skipTime(-1000);
            }
        });

        ffw.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                mediaPlayerComponent.mediaPlayer().controls().skipTime(1000);
            }
        });

        this.setContentPane(content);
        this.setVisible(true);
    }

    @Override
    public void eventDispatched(AWTEvent event) {
        if (event instanceof KeyEvent) {
            KeyEvent key = (KeyEvent) event;

            if (key.getID() == KeyEvent.KEY_PRESSED) {
                switch (key.getKeyChar()) {
                    case 'k' -> mediaPlayerComponent.mediaPlayer().controls().play();
                    case 'p' -> mediaPlayerComponent.mediaPlayer().controls().pause();
                    case 'l' -> mediaPlayerComponent.mediaPlayer().controls().skipTime(1000);
                    case 'j' -> mediaPlayerComponent.mediaPlayer().controls().skipTime(-1000);
                }
            }
        }
    }

    public void videoLoad(String pth) {
        mediaPlayerComponent.mediaPlayer().media().startPaused(pth);
    }

    public void run() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        }

        catch (Exception e) {
            System.out.println(e.toString());
        }

        init();
        setVisible(true);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new App(TITLE));
    }
}