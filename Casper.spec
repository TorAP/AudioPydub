# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['Combined_5.py'],
             pathex=['C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub'],
             binaries=[],
             datas=[],
             hiddenimports=['pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

a.datas += [('0.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\0.png', "DATA"),
('1.png', C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\1.png', "DATA"),
('2.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\2.png', "DATA"),
('3.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\3.png', "DATA"),
('4.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\4.png', "DATA"),
('5.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\5.png', "DATA"),
('-1.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\-1.png', "DATA"),
('-2.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\-2.png', "DATA"),
('-3.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\-3.png', "DATA"),
('-4.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\-4.png', "DATA"),
('-5.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\-5.png', "DATA"),
('alien.png','\alien.png', "DATA"),
('calib.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\calib.png', "DATA"),
('no.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\no.png', "DATA"),
('warning.png','C:\Users\caspe\OneDrive\Dokumenter\GitHub\AudioPydub\warning.png', "DATA")]

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Combined_5',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
