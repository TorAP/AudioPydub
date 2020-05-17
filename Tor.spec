# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['Combined_5.py'],
             pathex=['/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/Filters'],
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

a.datas += [('0.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/0.png', "DATA"),
('1.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/1.png', "DATA"),
('2.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/2.png', "DATA"),
('3.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/3.png', "DATA"),
('4.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/4.png', "DATA"),
('5.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/5.png', "DATA"),
('-1.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/-1.png', "DATA"),
('-2.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/-2.png', "DATA"),
('-3.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/-3.png', "DATA"),
('-4.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/-4.png', "DATA"),
('-5.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/-5.png', "DATA"),
('alien.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/alien.png', "DATA"),
('calib.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/calib.png', "DATA"),
('no.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/no.png', "DATA"),
('warning.png','/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/warning.png', "DATA")]

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
