# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['Combined_5.py'],
             pathex=['d:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters'],
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

a.datas += [('0.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\0.png', "DATA"),
('1.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\1.png', "DATA"),
('2.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\2.png', "DATA"),
('3.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\3.png', "DATA"),
('4.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\4.png', "DATA"),
('5.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\5.png', "DATA"),
('-1.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\-1.png', "DATA"),
('-2.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\-2.png', "DATA"),
('-3.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\-3.png', "DATA"),
('-4.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\-4.png', "DATA"),
('-5.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\-5.png', "DATA"),
('alien.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\alien.png', "DATA"),
('calib.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\calib.png', "DATA"),
('no.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\no.png', "DATA"),
('warning.png','D:\\Documents\\00_PERSONAL\\School\\MED4\\P4\\AudioPydub\\Filters\\warning.png', "DATA")]

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
