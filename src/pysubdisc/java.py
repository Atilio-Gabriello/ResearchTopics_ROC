
# Utility to determine the correct SubDisc JAR file for the JVM classpath
def extraClassPath():
  import os
  import re
  from . import __version__
  v = re.match("^[0-9]+\.[0-9]+\.(?P<sd>[0-9]+)$", __version__)
  if v is None:
    raise RuntimeError("Invalid package version")
  v = v.group('sd')
  if v == '9999':
    jar = 'subdisc-gui.jar'  # Use development JAR
  else:
    jar = f'subdisc-gui-r{v}.jar'  # Use versioned JAR
  jar = os.path.join(os.path.dirname(__file__), 'jars', jar)
  return jar


# Ensure the Java Virtual Machine is started with the correct classpath
def ensureJVMStarted():
  import jpype
  import jpype.imports
  if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[extraClassPath()])

  # Try to import a SubDisc class to raise an exception early if loading
  # SubDisc jar failed
  try:
    from nl.liacs.subdisc import Process
  except ModuleNotFoundError as e:
    # TODO: Python 3.11 has a cleaner approach using e.add_note()
    raise ModuleNotFoundError("Failed to import from SubDisc jar") from e


# Utility to temporarily redirect Java System.out/System.err (for suppressing JVM output)
def redirectSystemOutErr(f, *args, verbose=True, **kwargs):
  from java.lang import System
  from java.io import PrintStream, File
  # TODO: Consider capturing output to return to caller
  #from java.io import ByteArrayOutputStream

  if not verbose:
    import os
    oldOut = System.out
    oldErr = System.err
    System.out.flush()
    System.err.flush()
    # Redirect output to null device to suppress JVM output
    System.setOut(PrintStream(File(os.devnull)))
    System.setErr(PrintStream(File(os.devnull)))

  ret = f(*args, **kwargs)

  if not verbose:
    System.out.flush()
    System.err.flush()
    # Restore original output streams
    System.setOut(oldOut)
    System.setErr(oldErr)

  return ret
