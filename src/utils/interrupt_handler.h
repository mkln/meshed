#include "R.h"

inline static void chkIntFn(void *dummy) {
  R_CheckUserInterrupt();
}

// this will call the above in a top-level context so it won't longjmp-out of your context
inline bool checkInterrupt() {
  return (R_ToplevelExec(chkIntFn, NULL) == FALSE);
}
